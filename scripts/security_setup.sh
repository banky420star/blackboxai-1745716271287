#!/bin/bash

# Security Configuration Script
# This script implements security measures for the VPS

# Exit on any error
set -e

echo "Starting security configuration..."

# Update and install security packages
echo "Installing security packages..."
sudo apt update
sudo apt install -y \
    ufw \
    fail2ban \
    unattended-upgrades \
    rkhunter \
    logwatch \
    auditd \
    clamav

# Configure firewall
echo "Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable

# Configure fail2ban
echo "Configuring fail2ban..."
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF

sudo systemctl restart fail2ban

# Configure SSH
echo "Hardening SSH configuration..."
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
cat > /etc/ssh/sshd_config << EOF
# SSH Configuration
Port 22
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
AllowUsers trading_bot
EOF

# Setup automatic security updates
echo "Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};
Unattended-Upgrade::Package-Blacklist {
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::InstallOnShutdown "false";
Unattended-Upgrade::Mail "root";
Unattended-Upgrade::MailReport "on-change";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "02:00";
EOF

# Configure system auditing
echo "Setting up system auditing..."
cat > /etc/audit/rules.d/audit.rules << EOF
# Delete all existing rules
-D

# Buffer Size
-b 8192

# Failure Mode
-f 1

# Monitor file system mounts
-a always,exit -F arch=b64 -S mount -F auid>=1000 -F auid!=4294967295 -k mounts

# Monitor access to sensitive files
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes
-w /etc/sudoers -p wa -k sudoers_changes
-w /var/log/auth.log -p wa -k auth_log_changes

# Monitor trading bot files
-w /opt/trading_bot/src -p wa -k trading_bot_changes
-w /opt/trading_bot/config -p wa -k trading_bot_config_changes
EOF

sudo service auditd restart

# Setup API key security
echo "Setting up API key security..."
mkdir -p /opt/trading_bot/secrets
chmod 700 /opt/trading_bot/secrets

# Create API key storage with restricted permissions
cat > /opt/trading_bot/secrets/api_keys.env << EOF
# Trading Bot API Keys
# This file should be readable only by the trading bot process
EXCHANGE_API_KEY=
EXCHANGE_SECRET_KEY=
NEWS_API_KEY=
EOF

chmod 600 /opt/trading_bot/secrets/api_keys.env

# Setup security monitoring
echo "Setting up security monitoring..."
cat > /opt/trading_bot/scripts/security_check.sh << EOF
#!/bin/bash

# Check for failed login attempts
grep "Failed password" /var/log/auth.log | tail -n 10

# Check for unauthorized sudo usage
grep "sudo:" /var/log/auth.log | tail -n 10

# Check for modified system files
find /etc -mtime -1 -type f -ls

# Check running processes
ps aux | grep -v [p]s | grep -v [g]rep | grep -v [c]ron

# Check open ports
netstat -tuln

# Check disk usage
df -h

# Run rootkit check
rkhunter --check --skip-keypress
EOF

chmod +x /opt/trading_bot/scripts/security_check.sh

# Add security monitoring to cron
echo "0 * * * * /opt/trading_bot/scripts/security_check.sh > /var/log/trading_bot/security.log 2>&1" | crontab -l | sort -u | crontab -

# Setup log rotation
echo "Configuring log rotation..."
cat > /etc/logrotate.d/trading_bot << EOF
/var/log/trading_bot/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
}
EOF

echo "Security configuration completed successfully!"
echo "Please remember to:"
echo "1. Add your SSH public key to authorized_keys"
echo "2. Fill in the API keys in /opt/trading_bot/secrets/api_keys.env"
echo "3. Restart the SSH service: sudo systemctl restart sshd"
