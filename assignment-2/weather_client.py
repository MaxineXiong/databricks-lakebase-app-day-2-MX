"""
Client for the National Weather Service API.

The NWS API is free and public - no authentication required.
API documentation: https://www.weather.gov/documentation/services-web-api

Provides methods to:
- Resolve locations (lat/lon) to NWS grid points
- Fetch active weather alerts by state
- Fetch forecast data
- Normalize all data into standardized document records for ingestion
"""

import hashlib
from datetime import datetime, timezone
from dateutil import parser as date_parser
from typing import Any
import requests
from geopy.geocoders import ArcGIS

_DEFAULT_TIMEOUT = 30

class WeatherClient:
    """Client for the National Weather Service API with rate-limit friendly session."""

    def __init__(self, base_url: str = "https://api.weather.gov", timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # NWS API requests a User-Agent header
        self._session.headers.update(
            {
                "User-Agent": "(Databricks Weather Data Pipeline, contact@example.com)",
                "Accept": "application/geo+json",
            }
        )
        # Initialize geocoder for city->lat/lon conversion
        self._geocoder = ArcGIS()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request to the NWS API."""
        resp = self._session.get(
            f"{self.base_url}{path}", 
            params=params, 
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()
    
    def geocode_city(self, city: str, state: str = None) -> tuple[float, float] | None:
        """
        Convert city name to lat/lon coordinates using geopy.
        
        Args:
            city: City name (e.g., "Chicago")
            state: Optional 2-letter state code (e.g., "IL")
        
        Returns:
            Tuple of (lat, lon) or None if not found
        """
        try:
            query = f"{city}, {state}, USA" if state else f"{city}, USA"
            location = self._geocoder.geocode(query, timeout=self.timeout)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Geocoding failed for {query}: {e}")
        return None

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """
        Resolve a lat/lon pair to an NWS grid point.
        
        Args:
            lat: Latitude (decimal degrees)
            lon: Longitude (decimal degrees)
        
        Returns:
            Grid point data with properties.gridId, properties.gridX, properties.gridY
        """
        data = self.get(f"/points/{lat},{lon}")
        return data

    def get_active_alerts(self, state: str) -> list[dict]:
        """
        Fetch active weather alerts for a given state.
        
        Args:
            state: 2-letter state code (e.g., "CA", "NY")
        
        Returns:
            List of alert features from the NWS API
        """
        data = self.get("/alerts/active", params={"area": state})
        return data.get("features", [])

    def get_forecast(self, office: str, grid_x: int, grid_y: int) -> dict:
        """
        Fetch the regular forecast for a grid point.
        
        Args:
            office: NWS office ID (e.g., "TOP")
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
        
        Returns:
            Forecast data with periods containing detailed narrative text
        """
        data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
        return data

    def normalize_alert(self, alert_feature: dict, city: str | None, state: str | None, lat: float, lon: float) -> dict:
        """
        Normalize an NWS alert into a standardized document record.
        
        Args:
            alert_feature: Raw alert feature from NWS API
            city: City name or None if using lat/lon only
            state: State code or None
            lat: Latitude
            lon: Longitude
        
        Returns:
            Normalized document with required fields
        """
        props = alert_feature.get("properties", {})
        alert_id = props.get("id", "")
        
        # Combine description and instruction into narrative text
        description = props.get("description", "")
        instruction = props.get("instruction", "")
        narrative_parts = [p for p in [description, instruction] if p]
        narrative_text = "\n\n".join(narrative_parts)
        
        # Convert effective timestamp to UTC with Z suffix
        effective_at = props.get("effective")
        if effective_at:
            dt = date_parser.parse(effective_at)
            effective_at = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        return {
            "id": alert_id,
            "city": city,
            "state": state,
            "lat": lat,
            "lon": lon,
            "source_type": "alert",
            "headline": f"{props.get('headline', '')}",
            "narrative_text": narrative_text.strip(),
            "effective_at": effective_at,
            "payload": alert_feature,
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }

    def normalize_forecast_period(
        self,
        period: dict,
        city: str | None,
        state: str | None,
        lat: float,
        lon: float,
        office: str,
        grid_x: int,
        grid_y: int,
    ) -> dict:
        """
        Normalize a forecast period into a standardized document record.
        
        Args:
            period: Raw forecast period from NWS API
            city: City name or None if using lat/lon only
            state: State code or None
            lat: Latitude
            lon: Longitude
            office: NWS office ID
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
        
        Returns:
            Normalized document with required fields
        """
        # Create a stable ID based on location + period start time
        period_start = period.get("startTime", "")
        location_key = f"{lat}_{lon}_{period_start}"
        doc_id = hashlib.sha256(location_key.encode()).hexdigest()
        
        # Build headline from period name and conditions
        short_forecast = period.get("shortForecast", "")
        headline = f"{period.get('name', '')} - {short_forecast}"
        
        # Convert effective timestamp to UTC with Z suffix
        effective_at = period_start
        if effective_at:
            dt = date_parser.parse(effective_at)
            effective_at = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        return {
            "id": doc_id,
            "city": city,
            "state": state,
            "lat": lat,
            "lon": lon,
            "source_type": "forecast",
            "headline": headline.strip(),
            "narrative_text": period.get("detailedForecast", ""),
            "effective_at": effective_at,
            "payload": period,
            "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }

    def fetch_weather_data(
        self,
        city: str | None = None,
        state: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Fetch all weather data for a location and normalize it.
        
        Provide either city/state OR lat/lon (not both).
        
        Args:
            city: City name (e.g., "Chicago") - required if lat/lon not provided
            state: Optional 2-letter state code (e.g., "IL") - used for alerts and geocoding
            lat: Latitude in decimal degrees - alternative to city/state
            lon: Longitude in decimal degrees - alternative to city/state
            limit: Maximum number of documents to return per location (None = no limit)
        
        Returns:
            List of normalized document records (alerts + forecasts)
        
        Examples:
            >>> client = WeatherClient()
            >>> # Using city/state
            >>> docs = client.fetch_weather_data(city="Chicago", state="IL")
            >>> docs = client.fetch_weather_data(city="Austin", state="TX", limit=10)
            >>> # Using lat/lon
            >>> docs = client.fetch_weather_data(lat=41.8781, lon=-87.6298, state="IL", limit=5)
        """
        documents = []
        
        # Determine coordinates
        if city and state:
            # Geocode city to get coordinates
            coords = self.geocode_city(city, state)
            if not coords:
                print(f"Could not geocode the city and state: {city}, {state}")
                return []
            lat, lon = coords

            # Build location label for error messages
            location_label = f"{city}, {state}"
        else:
            print("Must provide both city and state")
            return []
        
        # 1. Get grid point
        try:
            grid_data = self.get_grid_point(lat, lon)
            props = grid_data.get("properties", {})
            office = props.get("gridId", "")
            grid_x = props.get("gridX", 0)
            grid_y = props.get("gridY", 0)
        except Exception as e:
            print(f"Failed to get grid point for {location_label} ({lat},{lon}): {e}")
            return documents
        
        # 2. Get active alerts (if state provided)
        try:
            alerts = self.get_active_alerts(state)
            for alert_feature in alerts:
                doc = self.normalize_alert(alert_feature, city, state, lat, lon)
                documents.append(doc)
        except Exception as e:
            print(f"Failed to get alerts for {state}: {e}")
        
        # 3. Get regular forecast
        try:
            forecast_data = self.get_forecast(office, grid_x, grid_y)
            periods = forecast_data.get("properties", {}).get("periods", [])
            for period in periods:
                doc = self.normalize_forecast_period(
                    period, city, state, lat, lon, office, grid_x, grid_y
                )
                documents.append(doc)
        except Exception as e:
            print(f"Failed to get forecast for {location_label}: {e}")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            documents = documents[:limit]
        
        return documents