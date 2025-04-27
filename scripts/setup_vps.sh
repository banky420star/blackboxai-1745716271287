#!/bin/bash

# VPS Setup and Configuration Script
# This script sets up the VPS environment for the trading bot

# Exit on any error
set -e

echo "Starting VPS setup..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    software-properties-common \
    curl \
    wget \
    htop \
    nginx \
    supervisor

# Create trading bot directory
echo "Creating trading bot directory..."
mkdir -p /opt/trading_bot
cd /opt/trading_bot

# Create virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup logging directory
echo "Setting up logging directory..."
mkdir -p /var/log/trading_bot
chmod 755 /var/log/trading_bot

# Setup supervisor configuration
echo "Configuring supervisor..."
cat > /etc/supervisor/conf.d/trading_bot.conf << EOF
[program:trading_bot]
directory=/opt/trading_bot
command=/opt/trading_bot/venv/bin/python src/main.py
user=root
autostart=true
autorestart=true
stderr_logfile=/var/log/trading_bot/error.log
stdout_logfile=/var/log/trading_bot/output.log
environment=
    PYTHONPATH="/opt/trading_bot",
    EXCHANGE_API_KEY="%(ENV_EXCHANGE_API_KEY)s",
    EXCHANGE_SECRET_KEY="%(ENV_EXCHANGE_SECRET_KEY)s"
EOF

# Reload supervisor
echo "Reloading supervisor..."
sudo supervisorctl reread
sudo supervisorctl update

# Setup monitoring
echo "Setting up monitoring..."
mkdir -p /opt/trading_bot/monitoring
cat > /opt/trading_bot/monitoring/check_bot.sh << EOF
#!/bin/bash

# Check if trading bot is running
if ! pgrep -f "python src/main.py" > /dev/null; then
    echo "Trading bot is not running. Restarting..."
    sudo supervisorctl restart trading_bot
fi

# Check disk space
DISK_USAGE=\$(df -h / | awk 'NR==2 {print \$5}' | sed 's/%//')
if [ \$DISK_USAGE -gt 90 ]; then
    echo "Warning: Disk usage is above 90%"
fi

# Check memory usage
FREE_MEM=\$(free | awk '/Mem:/ {print \$4}')
if [ \$FREE_MEM -lt 1000000 ]; then
    echo "Warning: Low memory available"
fi
EOF

chmod +x /opt/trading_bot/monitoring/check_bot.sh

# Add monitoring cron job
echo "*/5 * * * * /opt/trading_bot/monitoring/check_bot.sh >> /var/log/trading_bot/monitoring.log 2>&1" | crontab -

# Setup backup script
echo "Setting up backup system..."
mkdir -p /opt/trading_bot/backups
cat > /opt/trading_bot/scripts/backup.sh << EOF
#!/bin/bash

# Create backup directory with timestamp
BACKUP_DIR="/opt/trading_bot/backups/backup_\$(date +%Y%m%d_%H%M%S)"
mkdir -p \$BACKUP_DIR

# Backup trading bot files
cp -r /opt/trading_bot/src \$BACKUP_DIR/
cp -r /opt/trading_bot/models \$BACKUP_DIR/
cp -r /opt/trading_bot/config \$BACKUP_DIR/

# Backup logs
cp -r /var/log/trading_bot \$BACKUP_DIR/logs

# Compress backup
tar -czf \$BACKUP_DIR.tar.gz \$BACKUP_DIR
rm -rf \$BACKUP_DIR

# Remove backups older than 7 days
find /opt/trading_bot/backups -name "backup_*.tar.gz" -mtime +7 -delete
EOF

chmod +x /opt/trading_bot/scripts/backup.sh

# Add backup cron job
echo "0 0 * * * /opt/trading_bot/scripts/backup.sh" | crontab -l | sort -u | crontab -

echo "VPS setup completed successfully!"
