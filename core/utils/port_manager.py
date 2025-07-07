"""
Port Manager for Smart Pill Recognition System
Handles dynamic port allocation and server configuration
"""

import socket
import subprocess
import time
import psutil
from typing import List, Optional, Dict
import logging

class PortManager:
    """
    Manages port allocation and checks for server deployment
    """
    
    # Default ports to try
    DEFAULT_PORTS = [8501, 8502, 8503, 8504, 8505]
    RESTRICTED_PORTS = [8088, 8051, 22, 80, 443, 3306, 5432, 6379]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_port_available(self, port: int, host: str = "localhost") -> bool:
        """
        Check if a port is available for use
        
        Args:
            port: Port number to check
            host: Host to check (default: localhost)
            
        Returns:
            True if port is available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception as e:
            self.logger.warning(f"Error checking port {port}: {e}")
            return False
    
    def find_available_port(
        self, 
        preferred_ports: Optional[List[int]] = None,
        start_range: int = 8500,
        end_range: int = 9000
    ) -> Optional[int]:
        """
        Find an available port
        
        Args:
            preferred_ports: List of preferred ports to try first
            start_range: Start of port range to search
            end_range: End of port range to search
            
        Returns:
            Available port number or None if none found
        """
        ports_to_try = []
        
        # First try preferred ports
        if preferred_ports:
            ports_to_try.extend(preferred_ports)
        
        # Then try default ports
        ports_to_try.extend(self.DEFAULT_PORTS)
        
        # Finally try range
        ports_to_try.extend(range(start_range, end_range))
        
        for port in ports_to_try:
            if port in self.RESTRICTED_PORTS:
                continue
                
            if self.is_port_available(port):
                self.logger.info(f"Found available port: {port}")
                return port
        
        return None
    
    def get_port_info(self, port: int) -> Dict:
        """
        Get information about what's using a port
        
        Args:
            port: Port to check
            
        Returns:
            Dictionary with port information
        """
        info = {
            "port": port,
            "available": self.is_port_available(port),
            "process": None,
            "pid": None
        }
        
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    if conn.pid:
                        process = psutil.Process(conn.pid)
                        info["process"] = process.name()
                        info["pid"] = conn.pid
                    break
        except Exception as e:
            self.logger.warning(f"Error getting port info for {port}: {e}")
        
        return info
    
    def kill_process_on_port(self, port: int) -> bool:
        """
        Kill process using a specific port
        
        Args:
            port: Port number
            
        Returns:
            True if process was killed, False otherwise
        """
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.pid:
                    process = psutil.Process(conn.pid)
                    process.terminate()
                    time.sleep(2)
                    if process.is_running():
                        process.kill()
                    self.logger.info(f"Killed process {conn.pid} on port {port}")
                    return True
        except Exception as e:
            self.logger.error(f"Error killing process on port {port}: {e}")
        
        return False
    
    def check_server_constraints(self) -> Dict:
        """
        Check server constraints and restrictions
        
        Returns:
            Dictionary with constraint information
        """
        constraints = {
            "restricted_ports": self.RESTRICTED_PORTS,
            "available_ports": [],
            "recommendations": []
        }
        
        # Check which ports are available
        for port in range(8500, 9000):
            if self.is_port_available(port):
                constraints["available_ports"].append(port)
        
        # Add recommendations
        if not constraints["available_ports"]:
            constraints["recommendations"].append(
                "No ports available in range 8500-9000. Try using Docker with port mapping."
            )
        
        if 8088 in self.RESTRICTED_PORTS:
            constraints["recommendations"].append(
                "Port 8088 is restricted. Use alternative ports like 8501-8505."
            )
        
        if 8051 in self.RESTRICTED_PORTS:
            constraints["recommendations"].append(
                "Port 8051 is restricted. Use alternative ports like 8501-8505."
            )
        
        return constraints

def get_streamlit_port(preferred_port: int = 8501) -> int:
    """
    Get available port for Streamlit application
    
    Args:
        preferred_port: Preferred port to use
        
    Returns:
        Available port number
    """
    port_manager = PortManager()
    
    # Try preferred port first
    if port_manager.is_port_available(preferred_port):
        return preferred_port
    
    # Find alternative
    available_port = port_manager.find_available_port([preferred_port])
    
    if available_port is None:
        # Fall back to system-assigned port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', 0))
            return sock.getsockname()[1]
    
    return available_port

def check_docker_ports() -> List[int]:
    """
    Check which ports are available for Docker deployment
    
    Returns:
        List of available ports
    """
    port_manager = PortManager()
    available_ports = []
    
    # Check common Docker ports
    docker_ports = [8500, 8501, 8502, 8503, 8504, 8505, 8080, 8090]
    
    for port in docker_ports:
        if port_manager.is_port_available(port):
            available_ports.append(port)
    
    return available_ports

def setup_port_forwarding(internal_port: int, external_port: int) -> str:
    """
    Generate Docker port forwarding command
    
    Args:
        internal_port: Internal container port
        external_port: External host port
        
    Returns:
        Docker port mapping string
    """
    return f"-p {external_port}:{internal_port}"

if __name__ == "__main__":
    # Test port management
    port_manager = PortManager()
    
    print("🔍 Checking Port Availability...")
    print("=" * 40)
    
    # Check server constraints
    constraints = port_manager.check_server_constraints()
    print(f"Restricted ports: {constraints['restricted_ports']}")
    print(f"Available ports (sample): {constraints['available_ports'][:10]}")
    
    for rec in constraints["recommendations"]:
        print(f"💡 {rec}")
    
    # Test specific ports
    test_ports = [8088, 8051, 8501, 8502]
    print(f"\n🧪 Testing specific ports:")
    for port in test_ports:
        info = port_manager.get_port_info(port)
        status = "✅ Available" if info["available"] else f"❌ Used by {info['process']} (PID: {info['pid']})"
        print(f"  Port {port}: {status}")
    
    # Find recommended port
    recommended_port = get_streamlit_port()
    print(f"\n🚀 Recommended Streamlit port: {recommended_port}")
    
    # Check Docker ports
    docker_ports = check_docker_ports()
    print(f"🐳 Available Docker ports: {docker_ports[:5]}")
