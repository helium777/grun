import requests
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Type, Protocol
from urllib.parse import urljoin
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor
from .config import settings


class NotificationConfig(Protocol):
    """Protocol defining the interface for notification service configuration."""
    enabled: bool
    service_name: str


class BaseNotifier(ABC):
    """Base class for notification services with common functionality."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize the notifier with its configuration.
        
        Args:
            config: Configuration object implementing NotificationConfig protocol
        """
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    @abstractmethod
    def send(self, title: str, message: str) -> bool:
        """Send a notification with the given title and message.
        
        Args:
            title: Notification title
            message: Notification message content
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        pass
    
    def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> bool:
        """Make HTTP request with common error handling in a separate thread.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional arguments to pass to requests.request
            
        Returns:
            bool: True if request was successful, False otherwise
        """
        def _request():
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    timeout=5,
                    **kwargs
                )
                response.raise_for_status()
                return True
            except RequestException as e:
                print(f"Error sending notification: {e}")
                return False
        
        # Submit the request to the thread pool and return immediately
        self._executor.submit(_request)
        return True  # Return True immediately since we're not waiting for the result


class BarkNotifier(BaseNotifier):
    """Notifier implementation for Bark service (iOS notifications)."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize Bark notifier.
        
        Args:
            config: Configuration containing Bark service settings
        """
        super().__init__(config)
        self.base_url = config.server.rstrip("/")
        self.key = config.key
    
    def send(self, title: str, message: str) -> bool:
        """Send a notification using Bark service.
        
        Args:
            title: Notification title
            message: Notification message content
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.key:
            return False
        
        url = urljoin(self.base_url, f"{self.key}/{title}/{message}")
        return self._make_request("GET", url)


class TelegramNotifier(BaseNotifier):
    """Notifier implementation for Telegram service."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize Telegram notifier.
        
        Args:
            config: Configuration containing Telegram service settings
        """
        super().__init__(config)
        self.base_url = "https://api.telegram.org"
        self.bot_token = config.bot_token
        self.chat_id = config.chat_id
    
    def send(self, title: str, message: str) -> bool:
        """Send a notification using Telegram service.
        
        Args:
            title: Notification title
            message: Notification message content
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            return False
        
        url = urljoin(self.base_url, f"bot{self.bot_token}/sendMessage")
        payload = {
            "chat_id": self.chat_id,
            "text": f"*{title}*\n{message}",
            "parse_mode": "Markdown"
        }
        
        return self._make_request("POST", url, json=payload)


class SlackNotifier(BaseNotifier):
    """Notifier implementation for Slack service."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize Slack notifier.
        
        Args:
            config: Configuration containing Slack service settings
        """
        super().__init__(config)
        self.webhook_url = config.webhook_url
    
    def send(self, title: str, message: str) -> bool:
        """Send a notification using Slack service.
        
        Args:
            title: Notification title
            message: Notification message content
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        if not self.webhook_url:
            return False
        
        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": title,
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                }
            ]
        }
        
        return self._make_request("POST", self.webhook_url, json=payload)


class NullNotifier(BaseNotifier):
    """Null implementation of the Notifier interface (no-op)."""
    
    def send(self, title: str, message: str) -> bool:
        """Do nothing, return True.
        
        Args:
            title: Notification title (unused)
            message: Notification message content (unused)
            
        Returns:
            bool: Always returns True
        """
        return True


class NotifierRegistry:
    """Registry for managing notification service implementations."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self._notifiers: Dict[str, Type[BaseNotifier]] = {}
    
    def register(self, name: str, notifier_class: Type[BaseNotifier]) -> None:
        """Register a notifier class with the given name.
        
        Args:
            name: Unique identifier for the notifier
            notifier_class: Class implementing BaseNotifier
        """
        self._notifiers[name] = notifier_class
    
    def get_notifier(self, name: str, config: NotificationConfig) -> BaseNotifier:
        """Get a notifier instance for the given name and config.
        
        Args:
            name: Name of the notifier to retrieve
            config: Configuration for the notifier
            
        Returns:
            BaseNotifier: Instance of the requested notifier or NullNotifier if not found
        """
        notifier_class = self._notifiers.get(name)
        if not notifier_class:
            return NullNotifier(config)
        return notifier_class(config)


# Create global registry instance
registry = NotifierRegistry()

# Register built-in notifiers
registry.register("bark", BarkNotifier)
registry.register("telegram", TelegramNotifier)
registry.register("slack", SlackNotifier)
registry.register("none", NullNotifier)


def get_notifier() -> BaseNotifier:
    """Get a notifier instance based on current configuration.
    
    Returns:
        BaseNotifier: Configured notifier instance
    """
    notification_settings = getattr(settings.notification, settings.notification.service)
    return registry.get_notifier(
        settings.notification.service,
        notification_settings
    )
