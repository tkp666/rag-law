// 防止脚本重复执行的标志
if (window.LawRAGApp) {
  console.log("LawRAG App script already loaded, skipping duplicate initialization");
  // 如果已经初始化过，直接更新用户状态
  if (typeof updateUserStatus === 'function') {
    updateUserStatus();
  }
  exit;
}

// 设置标志，表明脚本已加载
window.LawRAGApp = true;

const askBtn = document.getElementById('askBtn');
const qEl = document.getElementById('q');
const statusEl = document.getElementById('status');
const answerArea = document.getElementById('answerArea');
const answerEl = document.getElementById('answer');
const retrievalsEl = document.getElementById('retrievals');
const topkEl = document.getElementById('topk');
const modelToggle = document.getElementById('modelToggle');
const modelLabel = document.getElementById('modelLabel');
const modelUsed = document.getElementById('modelUsed');
const loginBtn = document.getElementById('loginBtn');
const registerBtn = document.getElementById('registerBtn');
const usernameEl = document.getElementById('username');
const passwordEl = document.getElementById('password');
const userStatus = document.getElementById('userStatus');
const historyList = document.getElementById('historyList');
const noHistory = document.getElementById('noHistory');
const historySidebar = document.getElementById('historySidebar');
const closeSidebarBtn = document.getElementById('closeSidebarBtn');
const showSidebarBtn = document.getElementById('showSidebarBtn');

// 用于存储用户token和状态
let userToken = localStorage.getItem('userToken') || null;
let userId = localStorage.getItem('userId') || null;
let username = localStorage.getItem('username') || null;

// 防止重复绑定的标志
let eventListenersAdded = false;

// Update UI based on user status
function updateUserStatus() {
  if (userToken && username) {
    userStatus.textContent = `已登录: ${username}`;
    loginBtn.textContent = '登出';
    // 确保登出函数绑定
    loginBtn.onclick = logout;
    // 隐藏用户名和密码输入框，隐藏注册按钮，但保留登出按钮
    document.getElementById('username').style.display = 'none';
    document.getElementById('password').style.display = 'none';
    document.getElementById('registerBtn').style.display = 'none';
    loadHistory();
  } else {
    userStatus.textContent = '未登录';
    loginBtn.textContent = '登录';
    // 确保登录函数绑定
    loginBtn.onclick = handleLogin;
    // 显示用户名和密码输入框，显示注册按钮
    document.getElementById('username').style.display = 'inline-block';
    document.getElementById('password').style.display = 'inline-block';
    document.getElementById('registerBtn').style.display = 'inline-block';
    // 确保历史记录区域显示正确的提示
    historyList.innerHTML = '<p id="noHistory">登录后查看历史记录</p>';
  }
}

// Update model label based on toggle
if (!eventListenersAdded) {
  modelToggle.addEventListener('change', () => {
    if (modelToggle.checked) {
      modelLabel.textContent = '深度思考';
      modelUsed.textContent = '深度思考模式';
      modelUsed.style.backgroundColor = '#f39c12';
    } else {
      modelLabel.textContent = '基础模式';
      modelUsed.textContent = '基础模式';
      modelUsed.style.backgroundColor = '#3498db';
    }
  });
}

// Show message using status area instead of alert
function showMessage(message, type = 'error') {
  let icon, color;
  switch(type) {
    case 'success':
      icon = 'fa-check-circle';
      color = '#27ae60';
      break;
    case 'warning':
      icon = 'fa-exclamation-triangle';
      color = '#f39c12';
      break;
    default: // error
      icon = 'fa-exclamation-triangle';
      color = '#e74c3c';
  }
  statusEl.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
  statusEl.style.color = color;

  // 3秒后恢复原始状态
  setTimeout(() => {
    if (statusEl.innerHTML.includes('请求中') === false) {
      statusEl.innerHTML = '<i class="fas fa-check-circle"></i> 就绪';
      statusEl.style.color = '';
    }
  }, 3000);
}

// Handle login
async function handleLogin() {
  const inputUsername = usernameEl.value.trim();
  const password = passwordEl.value.trim();

  if (userStatus.textContent == '未登录' && (!inputUsername || !password)) {
    showMessage('请输入用户名和密码');
    return;
  }

  try {
    const response = await fetch('http://localhost:8001/api/auth/login/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken') || ''  // 如果需要CSRF token
      },
      body: JSON.stringify({ username: inputUsername, password })
    });

    if (!response.ok) {
      const error = await response.json();
      showMessage(`登录失败: ${error.error || response.statusText}`);
      return;
    }

    const data = await response.json();

    // Store user info - 现在使用真正的token
    userToken = data.token;
    userId = data.user_id;
    username = data.username;  // 全局变量赋值

    localStorage.setItem('userToken', userToken);
    localStorage.setItem('userId', userId);
    localStorage.setItem('username', username);

    updateUserStatus();

    // Clear form
    usernameEl.value = '';
    passwordEl.value = '';

    showMessage('登录成功', 'success');
  } catch (err) {
    console.error('Login error:', err);
    showMessage('登录请求失败: ' + err.message);
  }
}

// Handle registration
async function handleRegister() {
  const username = usernameEl.value.trim();
  const password = passwordEl.value.trim();

  if (!username || !password) {
    showMessage('请填写用户名和密码');
    return;
  }

  try {
    const response = await fetch('http://localhost:8001/api/auth/register/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken') || ''  // 如果需要CSRF token
      },
      body: JSON.stringify({ username, password })
    });

    if (!response.ok) {
      const error = await response.json();
      showMessage(`注册失败: ${error.error || response.statusText}`);
      return;
    }

    const data = await response.json();
    showMessage('注册成功，请登录', 'success');
  } catch (err) {
    console.error('Registration error:', err);
    showMessage('注册请求失败: ' + err.message);
  }
}

// Handle logout
function logout() {
  userToken = null;
  userId = null;
  username = null;

  localStorage.removeItem('userToken');
  localStorage.removeItem('userId');
  localStorage.removeItem('username');

  // 清空历史记录面板
  historyList.innerHTML = '<p id="noHistory">登录后查看历史记录</p>';

  // 清空主界面的显示内容
  qEl.value = '';
  answerEl.textContent = '';
  retrievalsEl.innerHTML = '';
  answerArea.style.display = 'none';

  updateUserStatus();
  showMessage('已登出', 'success');
}

// Load user history
async function loadHistory() {
  if (!userToken) {
    historyList.innerHTML = '<p id="noHistory">请先登录查看历史记录</p>';
    return;
  }

  try {
    // 直接通过FastAPI获取历史，FastAPI会代理请求到Django
    const response = await fetch(`/history?user_token=${userToken}`);

    if (!response.ok) {
      throw new Error(`服务器错误 ${response.status}`);
    }

    const data = await response.json();

    if (data.history && data.history.length > 0) {
      historyList.innerHTML = '';
      data.history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.dataset.id = item.id; // 存储历史记录ID

        historyItem.innerHTML = `
          <div class="history-question">${item.question}</div>
          <div class="history-answer">${item.answer.substring(0, 100)}${item.answer.length > 100 ? '...' : ''}</div>
          <div class="history-actions">
            <button class="delete-btn" onclick="deleteHistoryItem('${item.id}', event)">删除</button>
          </div>
        `;

        // 添加点击事件，用于填充问题和答案
        historyItem.addEventListener('click', function(e) {
          // 阻止删除按钮点击事件冒泡到历史项目上
          if (e.target.classList.contains('delete-btn')) {
            return;
          }
          loadHistoryItem(item);
        });

        historyList.appendChild(historyItem);
      });
    } else {
      historyList.innerHTML = '<p id="noHistory">暂无查询历史</p>';
    }
  } catch (err) {
    console.error('Load history error:', err);
    historyList.innerHTML = '<p id="noHistory">获取历史记录失败</p>';
  }
}

// 加载特定历史记录到主界面
function loadHistoryItem(item) {
  qEl.value = item.question;
  answerEl.textContent = item.answer;

  // 清空之前的检索结果并填充新的
  if (Array.isArray(item.retrievals)) {
    retrievalsEl.innerHTML = '';
    item.retrievals.forEach((r, i) => {
      const div = document.createElement('div');
      div.className = 'retrieval';

      const meta = document.createElement('div');
      meta.className = 'retrieval-meta';

      const sourceInfo = document.createElement('span');
      sourceInfo.className = 'source-info';
      sourceInfo.textContent = `${i+1}. ${r.source || '未知来源'}${r.article_no ? ' | ' + r.article_no : ''}`;

      const scoreInfo = document.createElement('span');
      scoreInfo.className = 'score-info';
      scoreInfo.textContent = `相关度: ${r.score !== null && r.score !== undefined ? Number(r.score).toFixed(3) : 'N/A'}`;

      meta.appendChild(sourceInfo);
      meta.appendChild(scoreInfo);

      const textp = document.createElement('div');
      textp.className = 'retrieval-text';
      textp.textContent = r.text || '';

      div.appendChild(meta);
      div.appendChild(textp);
      retrievalsEl.appendChild(div);
    });
  } else {
    // 如果历史记录没有检索结果，清空检索区域
    retrievalsEl.innerHTML = '';
  }

  // 显示答案区域
  answerArea.style.display = 'block';
}

// 删除历史记录
async function deleteHistoryItem(id, event) {
  event.stopPropagation(); // 阻止事件冒泡到历史记录项上

  if (!confirm('确定要删除这条记录吗？')) {
    return;
  }

  try {
    const response = await fetch(`/history/${id}/delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Token ${userToken}`  // 注意使用Token认证
      },
      body: JSON.stringify({})  // 发送空的JSON体
    });

    if (!response.ok) {
      throw new Error(`删除失败 ${response.status}`);
    }

    // 删除成功后刷新历史记录
    loadHistory();

    showMessage('记录已删除', 'success');
  } catch (err) {
    console.error('Delete history error:', err);
    showMessage('删除失败: ' + err.message, 'error');
  }
}

// 从cookie获取CSRF token的辅助函数
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

// Ask button click handler
async function handleAsk() {
  const question = qEl.value.trim();
  if (!question) {
    showMessage('请输入问题');
    return;
  }
  const top_k = parseInt(topkEl.value) || 3;
  const useReasoner = modelToggle.checked; // Whether to use deepseek-reasoner
  const model = useReasoner ? "deepseek-reasoner" : "deepseek-chat"; // Model to use

  // UI updates
  statusEl.innerHTML = '<i class="fas fa-spinner fa-spin spinner"></i> 请求中…';
  statusEl.style.color = '#f39c12';
  askBtn.disabled = true;
  answerArea.style.display = 'none';
  answerEl.textContent = '';
  retrievalsEl.innerHTML = '';

  try {
    const payload = {
      question,
      top_k,
      model  // Add model parameter to the request
    };

    // Add user token if user is logged in
    if (userToken) {
      payload.user_token = userToken;
    }

    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`服务器错误 ${resp.status}: ${txt}`);
    }

    const data = await resp.json();

    // Update answer
    answerEl.textContent = data.answer || '(无回答)';

    // Update retrievals
    if (Array.isArray(data.retrievals)) {
      retrievalsEl.innerHTML = '';
      data.retrievals.forEach((r, i) => {
        const div = document.createElement('div');
        div.className = 'retrieval';

        const meta = document.createElement('div');
        meta.className = 'retrieval-meta';

        const sourceInfo = document.createElement('span');
        sourceInfo.className = 'source-info';
        sourceInfo.textContent = `${i+1}. ${r.source || '未知来源'}${r.article_no ? ' | ' + r.article_no : ''}`;

        const scoreInfo = document.createElement('span');
        scoreInfo.className = 'score-info';
        scoreInfo.textContent = `相关度: ${r.score !== null && r.score !== undefined ? Number(r.score).toFixed(3) : 'N/A'}`;

        meta.appendChild(sourceInfo);
        meta.appendChild(scoreInfo);

        const textp = document.createElement('div');
        textp.className = 'retrieval-text';
        textp.textContent = r.text || '';

        div.appendChild(meta);
        div.appendChild(textp);
        retrievalsEl.appendChild(div);
      });
    }

    answerArea.style.display = 'block';
    showMessage('完成', 'success');
    statusEl.style.color = '#27ae60';

    // Reload history to show the new query if user is logged in
    if (userToken) {
      setTimeout(loadHistory, 1000); // Delay to allow backend to save the history
    }
  } catch (err) {
    console.error(err);
    showMessage('请求失败：' + err.message);
    statusEl.style.color = '#e74c3c';
  } finally {
    askBtn.disabled = false;
  }
}

// 切换侧边栏显示/隐藏
function toggleSidebar() {
  historySidebar.classList.toggle('hidden');
  if (historySidebar.classList.contains('hidden')) {
    // 侧边栏被隐藏，显示浮动按钮
    closeSidebarBtn.title = '隐藏历史记录';
    showSidebarBtn.style.display = 'flex';
  } else {
    // 侧边栏显示，隐藏浮动按钮
    closeSidebarBtn.title = '隐藏历史记录';
    showSidebarBtn.style.display = 'none';
  }
}

// Add event listeners only once
if (!eventListenersAdded) {
  askBtn.addEventListener('click', handleAsk);
  registerBtn.addEventListener('click', handleRegister);
  closeSidebarBtn.addEventListener('click', toggleSidebar);
  showSidebarBtn.addEventListener('click', function() {
    // 显示侧边栏并隐藏浮动按钮
    historySidebar.classList.remove('hidden');
    showSidebarBtn.style.display = 'none';
  });
  eventListenersAdded = true;
}

// Initialize UI
updateUserStatus();

// 暴露删除函数到全局，以便在HTML中使用
window.deleteHistoryItem = deleteHistoryItem;