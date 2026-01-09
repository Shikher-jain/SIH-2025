"""
Email utilities for notifications
"""
import smtplib
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending notifications"""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.EMAILS_FROM_EMAIL
        self.from_name = settings.EMAILS_FROM_NAME
        self.use_tls = settings.SMTP_TLS
    
    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """Send email to recipients"""
        
        if not self._is_configured():
            logger.warning("Email service not configured. Skipping email send.")
            return False
        
        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>" if self.from_name else self.from_email
            msg['To'] = ', '.join(to_emails)
            
            # Add text content
            if text_content:
                text_part = MimeText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {', '.join(to_emails)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_analysis_complete_notification(
        self,
        user_email: str,
        field_id: str,
        analysis_results: dict
    ) -> bool:
        """Send notification when analysis is complete"""
        
        subject = f"Spectral Analysis Complete - Field {field_id}"
        
        html_content = f"""
        <html>
        <body>
            <h2>Spectral Analysis Complete</h2>
            <p>Your spectral analysis for field <strong>{field_id}</strong> has been completed.</p>
            
            <h3>Analysis Summary:</h3>
            <ul>
                <li>Processing Time: {analysis_results.get('processing_time_seconds', 0):.1f} seconds</li>
                <li>Confidence Score: {analysis_results.get('confidence_score', 0):.2f}</li>
                <li>Risk Level: {analysis_results.get('risk_assessment', {}).get('risk_level', 'Unknown')}</li>
            </ul>
            
            <p>You can view the detailed results in your dashboard.</p>
            
            <p>Best regards,<br>Spectral Health Mapping Team</p>
        </body>
        </html>
        """
        
        text_content = f"""
        Spectral Analysis Complete
        
        Your spectral analysis for field {field_id} has been completed.
        
        Analysis Summary:
        - Processing Time: {analysis_results.get('processing_time_seconds', 0):.1f} seconds
        - Confidence Score: {analysis_results.get('confidence_score', 0):.2f}
        - Risk Level: {analysis_results.get('risk_assessment', {}).get('risk_level', 'Unknown')}
        
        You can view the detailed results in your dashboard.
        
        Best regards,
        Spectral Health Mapping Team
        """
        
        return self.send_email([user_email], subject, html_content, text_content)
    
    def send_alert_notification(
        self,
        user_email: str,
        field_id: str,
        alert_type: str,
        alert_message: str
    ) -> bool:
        """Send alert notification"""
        
        subject = f"Alert: {alert_type} - Field {field_id}"
        
        html_content = f"""
        <html>
        <body>
            <h2 style="color: #ff6b35;">Alert: {alert_type}</h2>
            <p>An alert has been detected for field <strong>{field_id}</strong>.</p>
            
            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px;">
                <strong>Alert Details:</strong><br>
                {alert_message}
            </div>
            
            <p>Please review the field analysis and take appropriate action if necessary.</p>
            
            <p>Best regards,<br>Spectral Health Mapping Team</p>
        </body>
        </html>
        """
        
        text_content = f"""
        Alert: {alert_type}
        
        An alert has been detected for field {field_id}.
        
        Alert Details:
        {alert_message}
        
        Please review the field analysis and take appropriate action if necessary.
        
        Best regards,
        Spectral Health Mapping Team
        """
        
        return self.send_email([user_email], subject, html_content, text_content)
    
    def _is_configured(self) -> bool:
        """Check if email service is properly configured"""
        return bool(
            self.smtp_host and 
            self.smtp_port and 
            self.from_email
        )


# Global email service instance
email_service = EmailService()