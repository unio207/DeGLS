"use client";

import { useCallback, useRef, useState } from "react";

export interface Fix {
  label: string;
  lat: number;
  lon: number;
}

export type PlaceStatus = "idle" | "locating" | "naming" | "done" | "error";

function coordText(lat: number, lon: number): string {
  return `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}

/**
 * Turns a GPS fix into a place a person would actually write in a field book.
 * BigDataCloud's client endpoint is keyless and CORS-open; if it is unreachable
 * (no bars in the middle of a field is the normal case) the coordinates stand
 * in, so the button never fails silently the way the old one did.
 */
async function reverseGeocode(lat: number, lon: number, signal: AbortSignal): Promise<string> {
  const url = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Reverse geocode failed (${res.status}).`);
  const data: {
    city?: string;
    locality?: string;
    principalSubdivision?: string;
    principalSubdivisionCode?: string;
    countryName?: string;
    countryCode?: string;
  } = await res.json();

  const place = data.city || data.locality;
  const region = data.principalSubdivisionCode?.split("-").pop() || data.principalSubdivision;
  const parts = [place, region].filter(Boolean);
  if (parts.length === 0 && data.countryName) parts.push(data.countryName);
  if (parts.length === 0) throw new Error("No place name for that position.");
  return parts.join(", ");
}

export function usePlace() {
  const [status, setStatus] = useState<PlaceStatus>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const locate = useCallback(async (): Promise<Fix | null> => {
    setMessage(null);

    if (typeof navigator === "undefined" || !navigator.geolocation) {
      setStatus("error");
      setMessage("This browser can't share a location. Type the field name instead.");
      return null;
    }

    setStatus("locating");

    let position: GeolocationPosition;
    try {
      position = await new Promise<GeolocationPosition>((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 12_000,
          maximumAge: 60_000,
        });
      });
    } catch (err) {
      const code = (err as GeolocationPositionError | undefined)?.code;
      setStatus("error");
      if (code === 1) {
        setMessage("Location is turned off for this site. Allow it in your browser settings, or type the field name.");
      } else if (code === 3) {
        setMessage("Couldn't get a fix in time. Try again in the open, or type the field name.");
      } else {
        setMessage("Couldn't read your position. Type the field name instead.");
      }
      return null;
    }

    const lat = position.coords.latitude;
    const lon = position.coords.longitude;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("naming");

    try {
      const label = await reverseGeocode(lat, lon, controller.signal);
      setStatus("done");
      return { label, lat, lon };
    } catch {
      setStatus("done");
      setMessage("No signal to look up a place name — using coordinates.");
      return { label: coordText(lat, lon), lat, lon };
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
    setMessage(null);
  }, []);

  return { status, message, locate, reset };
}

export { coordText };
