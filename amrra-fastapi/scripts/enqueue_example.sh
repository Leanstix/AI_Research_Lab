#!/usr/bin/env bash
curl -X POST http://localhost:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"question":"Effects of regular sexual intercourse on prostate cancer risk"}'