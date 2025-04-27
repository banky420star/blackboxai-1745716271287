#!/usr/bin/env python3
"""
VPS Management Script for Trading Bot
"""
import os
import sys
import psutil
import argparse
import subprocess
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vps_manager.log'),
        logging.StreamHandler()
    ]
)

class VPSManager:
    """Manages VPS operations for the trading bot"""
    
    def __init__(self):
        self.trading_bot_path = Path('/opt/trading_bot')
        self.log_path = Path('/var/log/trading_bot')
        self.backup_path = Path('/opt/trading_bot/backups')
        
    def check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            logging.info(f"System Resources:")
            logging.info(f"CPU Usage: {cpu_percent}%")
            logging.info(f"Memory Usage: {memory_percent}%")
            logging.info(f"Disk Usage: {disk_percent}%")
            
            # Check if any resource is above 90%
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                logging.warning("System resources are critically high!")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking system resources: {str(e)}")
            return False
    
    def check_trading_bot(self):
        """Check if trading bot is running"""
        try:
            # Look for the trading bot process
            bot_running = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'] and 'main.py' in ' '.join(proc.info['cmdline']):
                    bot_running = True
                    logging.info(f"Trading bot is running (PID: {proc.info['pid']})")
                    break
            
            if not bot_running:
                logging.warning("Trading bot is not running!")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking trading bot status: {str(e)}")
            return False
    
    def create_backup(self):
        """Create backup of trading bot files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_path / f"backup_{timestamp}"
            
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup trading bot files
            subprocess.run([
                'cp', '-r',
                str(self.trading_bot_path / 'src'),
                str(self.trading_bot_path / 'models'),
                str(self.trading_bot_path / 'config'),
                str(backup_dir)
            ])
            
            # Backup logs
            subprocess.run(['cp', '-r', str(self.log_path), str(backup_dir / 'logs')])
            
            # Create tar archive
            subprocess.run([
                'tar', 'czf',
                f"{backup_dir}.tar.gz",
                str(backup_dir)
            ])
            
            # Remove temporary backup directory
            subprocess.run(['rm', '-rf', str(backup_dir)])
            
            # Remove old backups (keep last 7 days)
            subprocess.run([
                'find', str(self.backup_path),
                '-name', 'backup_*.tar.gz',
                '-mtime', '+7',
                '-delete'
            ])
            
            logging.info(f"Backup created successfully: {backup_dir}.tar.gz")
            return True
            
        except Exception as e:
            logging.error(f"Error creating backup: {str(e)}")
            return False
    
    def restart_trading_bot(self):
        """Restart the trading bot"""
        try:
            # Stop existing bot process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'] and 'main.py' in ' '.join(proc.info['cmdline']):
                    proc.terminate()
                    proc.wait()
                    logging.info(f"Stopped trading bot process (PID: {proc.info['pid']})")
            
            # Start new bot process
            subprocess.Popen([
                'python3',
                str(self.trading_bot_path / 'src/main.py')
            ])
            
            logging.info("Trading bot restarted successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error restarting trading bot: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='VPS Management Script')
    parser.add_argument('--check', action='store_true', help='Check system resources and trading bot status')
    parser.add_argument('--backup', action='store_true', help='Create backup of trading bot files')
    parser.add_argument('--restart', action='store_true', help='Restart trading bot')
    
    args = parser.parse_args()
    manager = VPSManager()
    
    if args.check:
        resources_ok = manager.check_system_resources()
        bot_ok = manager.check_trading_bot()
        
        if not resources_ok or not bot_ok:
            sys.exit(1)
            
    elif args.backup:
        if not manager.create_backup():
            sys.exit(1)
            
    elif args.restart:
        if not manager.restart_trading_bot():
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
