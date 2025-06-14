import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import FileResponse
import uuid
# app/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- SQLAlchemy imports and DB setup ---
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column
from sqlalchemy import String, Float, DateTime
from datetime import datetime
from typing import Optional

DATABASE_URL = "sqlite+aiosqlite:///./weather.db"

engine = create_async_engine(DATABASE_URL, echo=False)
Base = declarative_base()
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

class WeatherQuery(Base):
    __tablename__ = "weather_queries"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    query_type: Mapped[str] = mapped_column(String)
    location: Mapped[str] = mapped_column(String)
    latitude: Mapped[float] = mapped_column(nullable=True)
    longitude: Mapped[float] = mapped_column(nullable=True)
    temperature: Mapped[float] = mapped_column(nullable=True)
    condition: Mapped[str] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)


load_dotenv()
# print("Loaded YOUTUBE_API_KEY:", os.getenv("YOUTUBE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

@app.get("/")
def root():
    return {"message": "Weather AI App is running!"}

import re

@app.get("/weather/")
async def get_weather(
    city: Optional[str] = Query(None, description="City name (e.g., 'Paris')"),
    zip_code: Optional[str] = Query(None, description="Zip/Postal code (e.g., '10001')"),
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude")
):
    params = {
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    elif zip_code:
        params["zip"] = zip_code
    elif city:
        params["q"] = city
    else:
        raise HTTPException(status_code=400, detail="You must provide either city, zip_code, or lat/lon.")

    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)

    if response.status_code != 200:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Location not found or API failed",
                "api_status": response.status_code,
                "api_response": response.text
            }
        )

    data = response.json()
    icon_code = data["weather"][0]["icon"]
    icon_url = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"

    weather_info = {
        "location": data.get("name"),
        "temperature (°C)": data["main"]["temp"],
        "condition": data["weather"][0]["description"],
        "humidity (%)": data["main"]["humidity"],
        "wind speed (m/s)": data["wind"]["speed"],
        "icon": icon_url,
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"]
    }

    return JSONResponse(content=weather_info)


# 5-day Forecast Endpoint
from collections import defaultdict
from datetime import datetime

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

@app.get("/forecast/")
async def get_forecast(
    city: Optional[str] = Query(None, description="City name (e.g., 'Paris')"),
    zip_code: Optional[str] = Query(None, description="Zip/Postal code (e.g., '10001')"),
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude")
):
    params = {
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon
    elif zip_code:
        params["zip"] = zip_code
    elif city:
        params["q"] = city
    else:
        raise HTTPException(status_code=400, detail="You must provide either city, zip_code, or lat/lon.")

    async with httpx.AsyncClient() as client:
        response = await client.get(FORECAST_URL, params=params)

    if response.status_code != 200:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Forecast not available",
                "api_status": response.status_code,
                "api_response": response.text
            }
        )

    data = response.json()
    grouped = defaultdict(list)

    for entry in data["list"]:
        dt_txt = entry["dt_txt"]
        date_str = dt_txt.split(" ")[0]
        grouped[date_str].append(entry)

    summarized = []

    for date, entries in grouped.items():
        temps = [e["main"]["temp"] for e in entries]
        conditions = [e["weather"][0]["description"] for e in entries]
        icons = [e["weather"][0]["icon"] for e in entries]

        summary = {
            "date": date,
            "temp_max": round(max(temps), 1),
            "temp_min": round(min(temps), 1),
            "condition": max(set(conditions), key=conditions.count),
            "icon": f"https://openweathermap.org/img/wn/{icons[0]}@2x.png"
        }
        summarized.append(summary)

    return JSONResponse(content={"location": data["city"]["name"], "forecast": summarized})


# --- DB startup event ---
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# --- Save weather query endpoint ---
@app.post("/save/")
async def save_query(
    query_type: str,
    location: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    temperature: Optional[float] = None,
    condition: Optional[str] = None,
):
    async with async_session() as session:
        new_entry = WeatherQuery(
            query_type=query_type,
            location=location,
            latitude=lat,
            longitude=lon,
            temperature=temperature,
            condition=condition,
        )
        session.add(new_entry)
        await session.commit()
        return {"message": "Query saved!"}


# --- Retrieve saved queries endpoint ---
@app.get("/saved/")
async def get_saved_queries():
    async with async_session() as session:
        result = await session.execute(
            WeatherQuery.__table__.select().order_by(WeatherQuery.timestamp.desc())
        )
        rows = result.fetchall()
        return [
            {
                "id": row.id,
                "type": row.query_type,
                "location": row.location,
                "temp": row.temperature,
                "condition": row.condition,
                "time": row.timestamp
            }
            for row in rows
        ]


# --- Update saved query ---
from fastapi import Path

@app.put("/update/{query_id}")
async def update_query(
    query_id: int = Path(..., description="ID of the query to update"),
    location: Optional[str] = None,
    temperature: Optional[float] = None,
    condition: Optional[str] = None
):
    async with async_session() as session:
        result = await session.get(WeatherQuery, query_id)
        if not result:
            raise HTTPException(status_code=404, detail="Query not found")

        if location:
            result.location = location
        if temperature:
            result.temperature = temperature
        if condition:
            result.condition = condition

        await session.commit()
        return {"message": "Query updated!"}


# --- Delete saved query ---
@app.delete("/delete/{query_id}")
async def delete_query(query_id: int = Path(..., description="ID of the query to delete")):
    async with async_session() as session:
        result = await session.get(WeatherQuery, query_id)
        if not result:
            raise HTTPException(status_code=404, detail="Query not found")

        await session.delete(result)
        await session.commit()
        return {"message": f"Query with ID {query_id} deleted"}


# --- YouTube Videos Endpoint ---
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

@app.get("/videos/")
async def get_youtube_videos(location: str = Query(..., description="Location to search weather videos for")):
    params = {
        "key": YOUTUBE_API_KEY,
        "q": f"weather in {location}",
        "part": "snippet",
        "maxResults": 3,
        "type": "video",
        "order": "relevance"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(YOUTUBE_SEARCH_URL, params=params)

    if response.status_code != 200:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to fetch YouTube videos",
                "status": response.status_code,
                "youtube_response": response.text
            }
        )

    videos = response.json().get("items", [])
    result = []
    for video in videos:
        vid = video["id"]["videoId"]
        title = video["snippet"]["title"]
        link = f"https://www.youtube.com/watch?v={vid}"
        result.append({"title": title, "url": link})

    return {"videos": result}


# --- Export saved data endpoint ---
from fastapi import Query

@app.get("/export/")
async def export_saved_data(format: str = Query("csv", enum=["csv", "json"])):
    async with async_session() as session:
        result = await session.execute(
            WeatherQuery.__table__.select().order_by(WeatherQuery.timestamp.desc())
        )
        rows = result.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="No saved data to export.")

        data = [
            {
                "id": row.id,
                "type": row.query_type,
                "location": row.location,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "temperature": row.temperature,
                "condition": row.condition,
                "timestamp": row.timestamp.isoformat()
            }
            for row in rows
        ]

        df = pd.DataFrame(data)
        filename = f"export_{uuid.uuid4().hex}.{format}"

        if format == "csv":
            df.to_csv(filename, index=False)
        elif format == "json":
            df.to_json(filename, orient="records", indent=2)

        return FileResponse(path=filename, filename=filename, media_type="application/octet-stream")
# Load model and encoders
model = joblib.load("weather_mood_model_v3.pkl")
weather_main_encoder = joblib.load("weather_main_encoder.pkl")
mood_encoder = joblib.load("weather_mood_encoder.pkl")

class WeatherFeatures(BaseModel):
    temp: float
    humidity: float
    wind_speed: float
    weather_main: str

@app.post("/predict-mood/")
async def predict_weather_mood(features: WeatherFeatures):
    try:
        input_df = pd.DataFrame([{
            "temp": features.temp,
            "humidity": features.humidity,
            "wind_speed": features.wind_speed,
            "weather_main": features.weather_main
        }])
        # One-hot encode 'weather_main'
        cat_encoded = weather_main_encoder.transform(input_df[["weather_main"]])
        cat_encoded_df = pd.DataFrame(
            cat_encoded, 
            columns=weather_main_encoder.get_feature_names_out(["weather_main"])
        )
        # Drop original and combine
        df_final = input_df.drop(columns=["weather_main"]).reset_index(drop=True)
        X = pd.concat([df_final, cat_encoded_df], axis=1)
        # Predict (returns encoded label)
        pred_encoded = model.predict(X)[0]
        # Decode to original label
        pred_label = mood_encoder.inverse_transform([pred_encoded])[0]
        return {"predicted_mood": pred_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))