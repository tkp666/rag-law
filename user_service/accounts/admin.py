from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, QueryHistory


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'created_at')
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('created_at', 'updated_at')}),
    )
    readonly_fields = ('created_at', 'updated_at')
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ('email',)}),
    )


@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'question', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('question', 'answer')
    readonly_fields = ('created_at',)
