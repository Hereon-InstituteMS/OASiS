/* OASiS WebUI — Alpine.js component.
 *
 * Manages:
 *  - Model / mode / MCP-toggle UI state
 *  - Single WebSocket to /ws/{sid}, JSON event protocol
 *  - File browser + on-click visualisation
 *  - Approve/reject buttons for plan-mode tool calls
 *  - Interactive parameter sliders
 *
 * Keep this dependency-light — only Alpine + fetch + native WebSocket.
 */
function oasisApp() {
  return {
    // ─── Config from backend
    models: [], mcpServers: [], modes: [],
    savedSessions: [],

    // ─── Active session state (mirrors the JSON on disk)
    session: { id: '', model: 'mock', mode: 'accept',
               mcp_servers: ['oasis'],
               events: [], tokens_in: 0, tokens_out: 0 },

    // ─── UI state
    ws: null, connected: false, status: 'idle',
    prompt: '', events: [],
    dir: { entries: [], rel: '' }, cwd: '/',
    viz: null, params: [],

    async init() {
      const [m, s, md] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/mcp_servers').then(r => r.json()),
        fetch('/api/modes').then(r => r.json()),
      ]);
      this.models = m.models;
      this.mcpServers = s.servers;
      this.modes = md.modes;
      this.session.model = m.default;
      this.session.mode = md.default;
      this.session.mcp_servers = s.servers
        .filter(x => x.default_on).map(x => x.id);

      await this.refreshSessionList();
      if (this.savedSessions.length > 0) {
        await this.connect(this.savedSessions[0].id);
      } else {
        await this.newSession();
      }
      this.loadFiles('');
    },

    // ─── Sessions ─────────────────────────────────────────────
    async refreshSessionList() {
      const r = await fetch('/api/sessions').then(r => r.json());
      this.savedSessions = r.sessions;
    },

    async newSession() {
      const r = await fetch('/api/sessions', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          model: this.session.model, mode: this.session.mode,
          mcp_servers: this.session.mcp_servers,
        }),
      }).then(r => r.json());
      this.savedSessions.unshift({ id: r.id, n_events: 0 });
      await this.connect(r.id);
    },

    async connect(sid) {
      if (this.ws) this.ws.close();
      this.session.id = sid;
      const res = await fetch(`/api/sessions/${sid}`);
      if (res.ok) {
        this.session = await res.json();
        this.events = this.session.events.slice();
      }
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      this.ws = new WebSocket(`${proto}://${window.location.host}/ws/${sid}`);
      this.ws.onopen = () => { this.connected = true; this.status = 'connected'; };
      this.ws.onclose = () => { this.connected = false; this.status = 'disconnected'; };
      this.ws.onerror = e => { this.status = 'error: ' + (e.message || ''); };
      this.ws.onmessage = ev => {
        try { this.onEvent(JSON.parse(ev.data)); }
        catch (e) { console.warn('bad msg', ev.data); }
      };
    },

    // ─── Inbound events ───────────────────────────────────────
    onEvent(e) {
      if (e.type === 'agent_chunk') {
        // append to last open agent_msg
        const last = this.events[this.events.length - 1];
        if (last && last.type === 'agent_msg' && last._open) {
          last.text = (last.text || '') + (e.text || '');
        } else {
          this.events.push({ type: 'agent_msg', text: e.text, _open: true });
        }
        this.scrollToBottom();
        return;
      }
      if (e.type === 'done') {
        const last = this.events[this.events.length - 1];
        if (last && last.type === 'agent_msg') last._open = false;
      }
      if (e.type === 'token_count') {
        this.session.tokens_in += e.input || 0;
        this.session.tokens_out += e.output || 0;
      }
      if (e.type === 'status' && e.session) {
        this.session = Object.assign(this.session, e.session);
        this.events = this.session.events.slice();
      } else if (e.type === 'status') {
        this.status = e.message;
      }
      this.events.push(e);
      this.scrollToBottom();
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const el = this.$refs.log;
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    // ─── Outbound ────────────────────────────────────────────
    send() {
      const text = this.prompt.trim();
      if (!text || !this.connected) return;
      this.ws.send(JSON.stringify({ type: 'prompt', text }));
      this.prompt = '';
    },

    approve(e) {
      e.decided = true;
      this.ws.send(JSON.stringify({ type: 'approve', call_id: e.call_id }));
    },
    reject(e) {
      e.decided = true;
      const reason = prompt('Reason for rejection?') || '';
      this.ws.send(JSON.stringify({ type: 'reject',
                                    call_id: e.call_id, reason }));
    },
    sendSet(type, payload) {
      if (!this.connected) return;
      this.ws.send(JSON.stringify(Object.assign({ type }, payload)));
    },
    setMode(m) {
      this.session.mode = m;
      this.sendSet('set_mode', { mode: m });
    },
    toggleMcp(id) {
      const set = new Set(this.session.mcp_servers);
      set.has(id) ? set.delete(id) : set.add(id);
      this.session.mcp_servers = [...set];
      this.sendSet('set_mcp', { servers: this.session.mcp_servers });
    },
    restart() {
      if (!confirm('Restart this session?')) return;
      this.events = [];
      this.session.tokens_in = 0; this.session.tokens_out = 0;
      this.sendSet('restart', {});
    },

    // ─── Files & viz ─────────────────────────────────────────
    async loadFiles(rel) {
      const r = await fetch('/api/files?rel=' + encodeURIComponent(rel))
        .then(r => r.json());
      this.dir = r;
      this.cwd = r.rel ? '/' + r.rel : '/';
    },

    async onClick(e) {
      if (e.is_dir) {
        this.loadFiles(e.rel_path);
        return;
      }
      const r = await fetch('/api/viz?rel=' + encodeURIComponent(e.rel_path))
        .then(r => r.json());
      this.viz = r;
      if (r.kind === 'table' && r.plot) {
        this.$nextTick(() => {
          Plotly.newPlot('plotDiv', r.plot.data, r.plot.layout,
                         { displayModeBar: false, responsive: true });
        });
      }
      if (r.kind === 'vtk') {
        // vtk.js HTTPDataAccessHelper loads the URL; light wiring only.
        this.$nextTick(() => this.renderVtk(r.url));
      }
      if (e.kind === 'text' && e.name.endsWith('.py')) {
        const pr = await fetch('/api/extract_params?rel='
                               + encodeURIComponent(e.rel_path))
          .then(r => r.json());
        this.params = pr.params;
      }
    },

    async renderVtk(url) {
      const root = document.getElementById('vtkRoot');
      if (!root || typeof vtk === 'undefined') return;
      root.innerHTML = '<div class="text-slate-500 p-2">' +
        'vtk.js render scaffold (extend in renderVtk in app.js)</div>';
    },

    rerunWithParams() {
      const summary = this.params
        .map(p => `${p.name} = ${p.value}`).join('; ');
      this.prompt = 'Re-run the previous script with these parameter ' +
        'changes and report the result file: ' + summary;
      this.send();
    },

    // ─── Helpers ─────────────────────────────────────────────
    eventClass(e) {
      const m = {
        user_msg: 'border-emerald-700 bg-emerald-900/30',
        agent_msg: 'border-slate-700 bg-slate-800/50',
        agent_chunk: 'border-slate-700 bg-slate-800/50',
        tool_call_pending: 'border-amber-700 bg-amber-900/20',
        tool_call_executing: 'border-amber-700 bg-amber-900/20',
        tool_result: 'border-slate-700 bg-slate-900',
        subagent_spawned: 'border-violet-700 bg-violet-900/30',
        subagent_returned: 'border-violet-700 bg-violet-900/30',
        token_count: 'border-slate-800 bg-slate-900 text-xs',
        error: 'border-rose-700 bg-rose-900/30',
        done: 'border-emerald-700 bg-emerald-900/20',
        status: 'border-slate-800 bg-slate-900 text-xs text-slate-400',
      };
      return m[e.type] || 'border-slate-700 bg-slate-800/50';
    },
    formatEvent(e) {
      if (e.type === 'agent_msg' || e.type === 'user_msg') return e.text || '';
      if (e.type === 'tool_call_pending' || e.type === 'tool_call_executing') {
        return `${e.tool} (${JSON.stringify(e.args || {}).slice(0, 200)})`;
      }
      if (e.type === 'tool_result') return (e.result || '').slice(0, 800);
      if (e.type === 'subagent_spawned') {
        return `${e.role}: ${e.task}\n[context] ${e.context || ''}`;
      }
      if (e.type === 'subagent_returned') return (e.result || '').slice(0, 800);
      if (e.type === 'token_count') return `in=${e.input}  out=${e.output}`;
      const { type, ...rest } = e;
      return JSON.stringify(rest).slice(0, 400);
    },
    kindIcon(k) {
      return { dir: '📁', vtk: '🌐', hdf: '📦', image: '🖼',
               table: '📊', json: '{ }', yaml: '⚙', mesh: '🔲',
               text: '📄', binary: '⬛' }[k] || '·';
    },
    humanSize(n) {
      if (n == null) return '';
      if (n < 1024) return n + ' B';
      if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
      return (n / 1024 / 1024).toFixed(1) + ' MB';
    },
  };
}
