from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


class CustomUser(AbstractUser):
    email = models.EmailField(blank=True, unique=True)  # 邮箱不再是必须唯一字段
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'username'  # 使用用户名作为登录字段
    REQUIRED_FIELDS = ['email']  # 邮箱作为额外需要的字段

    def __str__(self):
        return self.username


class QueryHistory(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='queries')
    question = models.TextField()
    answer = models.TextField()
    source_info = models.JSONField(null=True, blank=True)  # 存储来源信息
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '查询历史'
        verbose_name_plural = '查询历史'

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}..."