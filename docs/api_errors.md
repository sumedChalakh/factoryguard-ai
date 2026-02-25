# API Error Handling

This API now returns JSON errors by default (production-safe behavior).

## Debug mode

- Default: debug is **off**.
- Enable debug traceback pages locally by setting `FLASK_DEBUG=1` before starting the app.

## Error response format

All handled errors return JSON in this shape:

```json
{
  "error": "<error type>",
  "message": "<human-readable message>"
}
```

## Common responses

### 400 Bad Request

Example (`/predict` or `/explain` with empty JSON body):

```json
{
  "error": "Empty JSON payload"
}
```

### 500 Internal Server Error

In production mode (`FLASK_DEBUG` not set or set to `0`):

```json
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```

In debug mode (`FLASK_DEBUG=1`), Flask debugger behavior is preserved for easier local debugging.

## PowerShell note

`Invoke-RestMethod` throws when HTTP status is not 2xx, even if the API returns valid JSON.
Use `try/catch` to inspect the body:

```powershell
try {
  Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body '{}'
} catch {
  $_.ErrorDetails.Message
}
```
