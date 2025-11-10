from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.middleware.csrf import get_token
from .models import QueryHistory
import json

User = get_user_model()


@csrf_exempt
def get_csrf_token(request):
    """获取CSRF token"""
    token = get_token(request)
    return JsonResponse({'csrfToken': token})


@csrf_exempt
@require_http_methods(["POST"])
def register(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return JsonResponse({'error': '请提供用户名和密码'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': '用户名已存在'}, status=400)

        # 如果没有提供邮箱，则使用默认邮箱格式
        email = f"{username}@default.com"
        if 'email' in data and data['email']:
            email = data['email']

        user = User.objects.create_user(username=username, email=email, password=password)
        # 为新用户创建token
        token, created = Token.objects.get_or_create(user=user)

        return JsonResponse({
            'message': '注册成功',
            'user_id': user.id,
            'username': user.username,
            'email': user.email,
            'token': token.key
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def login_view(request):
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            # 获取或创建用户token
            token, created = Token.objects.get_or_create(user=user)
            return JsonResponse({
                'message': '登录成功',
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'token': token.key
            })
        else:
            return JsonResponse({'error': '用户名或密码错误'}, status=401)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_history(request):
    try:
        user = request.user
        history = QueryHistory.objects.filter(user=user).order_by('-created_at')[:20]

        history_data = []
        for h in history:
            source_info_data = h.source_info
            # 确保 source_info 是一个字典
            if isinstance(source_info_data, str):
                try:
                    source_info_data = json.loads(source_info_data)
                except json.JSONDecodeError:
                    source_info_data = {}  # 如果解析失败，则使用空字典

            history_data.append({
                'id': h.id,
                'question': h.question,
                'answer': h.answer,
                'retrievals': source_info_data.get('retrievals', []),  # 提取retrievals
                'created_at': h.created_at.isoformat(),
            })

        return Response({'history': history_data})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def save_query(request):
    try:
        user = request.user
        data = request.data  # 使用DRF的request.data而不是request.body

        query_history = QueryHistory.objects.create(
            user=user,
            question=data['question'],
            answer=data['answer'],
            source_info=data.get('source_info', {})
        )

        return Response({
            'message': '查询历史保存成功',
            'id': query_history.id
        })
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
@csrf_exempt
def delete_history(request, history_id):
    try:
        user = request.user
        # 查找对应用户和ID的历史记录
        history_item = QueryHistory.objects.get(id=history_id, user=user)
        history_item.delete()
        
        return Response({
            'message': '查询历史删除成功'
        })
    except QueryHistory.DoesNotExist:
        return Response({'error': '历史记录不存在或无权限删除'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)