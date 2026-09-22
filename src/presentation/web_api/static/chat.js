/**
 * MireaChat — виджет чата для главной страницы и ЛКС.
 *
 * Использование:
 *   const chat = new MireaChat({ userContext: { course: 1, faculty: 'ИРТС' } });
 *   chat.mount();   // добавляет FAB + панель в document.body
 *
 * userContext — опционален. Если передан, добавляется к запросу на /api/ask
 * и отображается в заголовке панели (персонализированный режим).
 */

class MireaChat {
  constructor({ userContext = null, showPrivacyBanner = false } = {}) {
    this.userContext = userContext;
    this.showPrivacyBanner = showPrivacyBanner;
    this.isOpen = false;
    this.isLoading = false;
    this._privacyAck = sessionStorage.getItem('mirea_privacy_ack') === '1';
  }

  mount() {
    this._buildDOM();
    this._attachEvents();
    this._addWelcomeMessage();
  }

  _buildDOM() {
    // FAB button
    this.fab = document.createElement('button');
    this.fab.className = 'chat-fab';
    this.fab.setAttribute('aria-label', 'Открыть помощника');
    this.fab.innerHTML = '💬';

    // Panel
    this.panel = document.createElement('div');
    this.panel.className = 'chat-panel hidden';

    const personalized = this.userContext && (this.userContext.course || this.userContext.faculty);
    const subtitle = personalized ? 'Персонализированный режим' : 'Онлайн · отвечаю 24/7';

    this.panel.innerHTML = `
      <div class="chat-header">
        <div class="chat-header-info">
          <div class="chat-status-dot"></div>
          <div>
            <div class="chat-title">Помощник МИРЭА</div>
            <div class="chat-subtitle">${subtitle}</div>
          </div>
        </div>
        <button class="chat-close" aria-label="Закрыть">✕</button>
      </div>
      <div class="chat-messages" id="chat-messages"></div>
      <div class="quick-questions" id="quick-questions"></div>
      <div class="chat-input-area">
        <textarea class="chat-input" placeholder="Задайте вопрос..." rows="1" id="chat-input"></textarea>
        <button class="chat-send" id="chat-send">Отправить</button>
      </div>
    `;

    // Privacy banner (LCS only, once per session)
    if (this.showPrivacyBanner && !this._privacyAck) {
      const banner = document.createElement('div');
      banner.className = 'chat-privacy';
      banner.id = 'privacy-banner';
      banner.innerHTML = `
        <span>Бот видит данные вашего профиля для точных ответов.
          <a href="#" id="privacy-settings-link">Настроить приватность</a>
        </span>
        <button class="privacy-dismiss" aria-label="Закрыть">×</button>
      `;
      // Insert before input area
      this.panel.querySelector('.chat-input-area').before(banner);
    }

    document.body.appendChild(this.fab);
    document.body.appendChild(this.panel);

    this.messagesEl = this.panel.querySelector('#chat-messages');
    this.inputEl    = this.panel.querySelector('#chat-input');
    this.sendBtn    = this.panel.querySelector('#chat-send');
    this.quickEl    = this.panel.querySelector('#quick-questions');
  }

  _attachEvents() {
    this.fab.addEventListener('click', () => this.toggle());
    this.panel.querySelector('.chat-close').addEventListener('click', () => this.close());

    this.sendBtn.addEventListener('click', () => this._handleSend());
    this.inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._handleSend(); }
    });

    // Auto-resize textarea
    this.inputEl.addEventListener('input', () => {
      this.inputEl.style.height = 'auto';
      this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 96) + 'px';
    });

    // Privacy banner
    const banner = this.panel.querySelector('#privacy-banner');
    if (banner) {
      banner.querySelector('.privacy-dismiss').addEventListener('click', () => {
        sessionStorage.setItem('mirea_privacy_ack', '1');
        banner.remove();
      });
      const settingsLink = banner.querySelector('#privacy-settings-link');
      if (settingsLink) {
        settingsLink.addEventListener('click', (e) => {
          e.preventDefault();
          this._showPrivacySettings();
        });
      }
    }
  }

  _addWelcomeMessage() {
    const personalized = this.userContext && (this.userContext.course || this.userContext.faculty);
    let text;
    if (personalized) {
      const parts = [];
      if (this.userContext.course) parts.push(`${this.userContext.course} курс`);
      if (this.userContext.faculty) parts.push(this.userContext.faculty);
      text = `Привет! Я вижу, что вы студент (${parts.join(', ')}). Задайте вопрос — отвечу с учётом вашего профиля.`;
    } else {
      text = 'Привет! Я помощник МИРЭА. Задайте вопрос про поступление, учёбу, документы или жизнь в университете.';
    }
    this._appendMessage('bot', text);
    this._showQuickQuestions();
  }

  _showQuickQuestions() {
    const questions = this.userContext?.course
      ? ['Какие стипендии мне доступны?', 'Как оформить рассрочку?', 'Условия общежития']
      : ['Минимальные баллы ЕГЭ', 'Как подать на общежитие?', 'Стоимость обучения'];

    this.quickEl.innerHTML = '';
    questions.forEach(q => {
      const btn = document.createElement('button');
      btn.className = 'quick-btn';
      btn.textContent = q;
      btn.addEventListener('click', () => {
        this.quickEl.innerHTML = '';
        this._sendMessage(q);
      });
      this.quickEl.appendChild(btn);
    });
  }

  toggle() {
    this.isOpen ? this.close() : this.open();
  }

  open() {
    this.isOpen = true;
    this.panel.classList.remove('hidden');
    this.fab.innerHTML = '✕';
    this.inputEl.focus();
  }

  close() {
    this.isOpen = false;
    this.panel.classList.add('hidden');
    this.fab.innerHTML = '💬';
  }

  _handleSend() {
    const text = this.inputEl.value.trim();
    if (!text || this.isLoading) return;
    this.inputEl.value = '';
    this.inputEl.style.height = 'auto';
    this.quickEl.innerHTML = '';
    this._sendMessage(text);
  }

  async _sendMessage(text) {
    this._appendMessage('user', text);
    this._setLoading(true);
    const typingId = this._appendTyping();

    try {
      const body = { question: text };
      if (this.userContext) body.user_context = this.userContext;

      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      this._removeTyping(typingId);
      this._appendMessage('bot', data.answer, data.needs_human_fallback);
    } catch (err) {
      this._removeTyping(typingId);
      this._appendMessage('bot', 'Произошла ошибка при обращении к серверу. Попробуйте позже.', true);
    } finally {
      this._setLoading(false);
    }
  }

  _appendMessage(role, text, isFallback = false) {
    const msg = document.createElement('div');
    msg.className = `msg ${role}${isFallback ? ' fallback' : ''}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;

    const time = document.createElement('div');
    time.className = 'msg-time';
    time.textContent = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });

    msg.appendChild(bubble);
    msg.appendChild(time);
    this.messagesEl.appendChild(msg);
    this._scrollToBottom();
    return msg;
  }

  _appendTyping() {
    const id = `typing-${Date.now()}`;
    const msg = document.createElement('div');
    msg.className = 'msg bot';
    msg.id = id;
    msg.innerHTML = `<div class="msg-bubble"><div class="typing-dots">
      <span></span><span></span><span></span>
    </div></div>`;
    this.messagesEl.appendChild(msg);
    this._scrollToBottom();
    return id;
  }

  _removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  _scrollToBottom() {
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  _setLoading(state) {
    this.isLoading = state;
    this.sendBtn.disabled = state;
    this.inputEl.disabled = state;
  }

  _showPrivacySettings() {
    // Simple inline settings overlay
    const existing = document.getElementById('privacy-overlay');
    if (existing) return;

    const overlay = document.createElement('div');
    overlay.id = 'privacy-overlay';
    overlay.style.cssText = `
      position:absolute; inset:0; background:rgba(0,0,0,.5);
      display:flex; align-items:center; justify-content:center;
      z-index:10; border-radius:16px;
    `;
    overlay.innerHTML = `
      <div style="background:#fff;border-radius:10px;padding:24px;width:300px;font-size:14px;">
        <div style="font-weight:700;font-size:16px;margin-bottom:16px;color:#1c2951">
          Настройки приватности
        </div>
        <label style="display:flex;gap:10px;margin-bottom:12px;cursor:pointer;">
          <input type="checkbox" checked id="pref-course"> Курс и факультет
        </label>
        <label style="display:flex;gap:10px;margin-bottom:20px;cursor:pointer;">
          <input type="checkbox" checked id="pref-finance"> Финансовый статус
        </label>
        <div style="font-size:12px;color:#6b7280;margin-bottom:20px;">
          Данные используются только для персонализации ответов и не передаются третьим лицам.
        </div>
        <button id="privacy-save" style="
          background:#c8161d;color:#fff;border:none;border-radius:8px;
          padding:9px 20px;font-size:14px;font-weight:600;cursor:pointer;width:100%;
        ">Сохранить</button>
      </div>
    `;
    this.panel.style.position = 'relative';
    this.panel.appendChild(overlay);
    overlay.querySelector('#privacy-save').addEventListener('click', () => {
      sessionStorage.setItem('mirea_privacy_ack', '1');
      overlay.remove();
      const banner = document.getElementById('privacy-banner');
      if (banner) banner.remove();
    });
  }
}
