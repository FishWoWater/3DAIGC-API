
### Logging and Monitoring
The API provides comprehensive logging capabilities with multiple endpoints for monitoring and debugging:

```bash
# View recent log entries
curl -X GET "http://localhost:7842/api/v1/system/logs?lines=100&level=INFO"

# List all log files
curl -X GET "http://localhost:7842/api/v1/system/logs/files"

# Get specific log file contents
curl -X GET "http://localhost:7842/api/v1/system/logs/files/app.log?lines=50"

# View logging configuration
curl -X GET "http://localhost:7842/api/v1/system/logs/config"

# Generate test log entries for verification
curl -X POST "http://localhost:7842/api/v1/system/logs/test?level=INFO&count=5"

# Update logging level dynamically
curl -X PUT "http://localhost:7842/api/v1/system/logs/level?logger_name=root&level=DEBUG"

# Filter logs by logger name
curl -X GET "http://localhost:7842/api/v1/system/logs?logger_name=api&level=ERROR"
```

**Available Log Files:**
- `app.log` - Main application logs with detailed information
- `api.log` - API-specific logs (requests, responses, etc.)
- `error.log` - Error-level logs from all components
- `scheduler.log` - GPU scheduler and model management logs

**Log Filtering Options:**
- `lines` - Number of recent lines to return (default: 100)
- `level` - Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logger_name` - Filter by logger name (partial match)
- `since` - Filter logs since ISO timestamp
