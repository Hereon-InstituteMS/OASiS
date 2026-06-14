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
    editMode: false, saveStatus: '',
    vtkState: null, vtkRange: [0, 1],

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
      this.editMode = false;
      this.saveStatus = '';
      this.viz = null;
      if (e.is_dir) {
        this.loadFiles(e.rel_path);
        return;
      }
      this.viz = { kind: 'pending', rel: e.rel_path };
      const r = await fetch('/api/viz?rel=' + encodeURIComponent(e.rel_path))
        .then(r => r.json());
      r.rel = e.rel_path;
      this.viz = r;
      if (r.kind === 'table' && r.plot) {
        this.$nextTick(() => {
          // r.plot.config carries editable/edits/toImageButtonOptions
          // from the backend so axes & titles are click-editable and
          // the toolbar exposes PNG/SVG export.
          Plotly.newPlot('plotDiv', r.plot.data, r.plot.layout,
                         r.plot.config || {responsive: true});
        });
      }
      if (r.kind === 'vtk') {
        this.$nextTick(() => this.renderVtk(r.url));
      }
      if (e.kind === 'text' && e.name.endsWith('.py')) {
        const pr = await fetch('/api/extract_params?rel='
                               + encodeURIComponent(e.rel_path))
          .then(r => r.json());
        this.params = pr.params;
      } else {
        this.params = [];
      }
    },

    // ─── File save (text/JSON/YAML/script editor) ───
    async saveFile() {
      if (!this.viz || !this.viz.rel || this.viz.kind !== 'text') return;
      this.saveStatus = 'saving…';
      try {
        const r = await fetch('/api/file', {
          method: 'POST',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify({rel: this.viz.rel,
                                content: this.viz.text}),
        });
        const js = await r.json();
        if (js.ok) {
          this.saveStatus = 'saved (' + js.bytes + ' B)';
          setTimeout(() => { this.saveStatus = ''; }, 4000);
          this.editMode = false;
        } else {
          this.saveStatus = 'error: ' + (js.detail || js.error || 'unknown');
        }
      } catch (e) {
        this.saveStatus = 'error: ' + e.message;
      }
    },

    // ─── Plot export ───
    exportPlot(format) {
      const div = document.getElementById('plotDiv');
      if (!div || typeof Plotly === 'undefined') return;
      const name = (this.viz && this.viz.rel)
        ? this.viz.rel.split('/').pop().replace(/\.\w+$/, '') : 'plot';
      Plotly.downloadImage(div, {format, filename: name, scale: 2});
    },

    // ─── VTK rendering (real, vtk.js HTTPDataAccessHelper) ───
    async renderVtk(url) {
      const root = document.getElementById('vtkRoot');
      if (!root || typeof vtk === 'undefined') {
        if (root) root.innerHTML = '<div class="text-slate-500 p-2">' +
          'vtk.js library failed to load from CDN; check network.</div>';
        return;
      }
      root.innerHTML = '';
      const fsContainer = vtk.Rendering.Misc.vtkFullScreenRenderWindow
        ? vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
            rootContainer: root, background: [0.06, 0.09, 0.16],
          }) : null;
      if (!fsContainer) return;
      const renderer = fsContainer.getRenderer();
      const renderWindow = fsContainer.getRenderWindow();
      const reader = vtk.IO.XML.vtkXMLPolyDataReader
        ? vtk.IO.XML.vtkXMLPolyDataReader.newInstance() : null;
      if (!reader) {
        root.innerHTML = '<div class="text-slate-500 p-2">' +
          'vtk.js modules missing — load full vtk.js bundle.</div>';
        return;
      }
      try {
        const resp = await fetch(url);
        const buf = await resp.arrayBuffer();
        reader.parseAsArrayBuffer(buf);
      } catch (e) {
        root.innerHTML = '<div class="text-rose-300 p-2">' +
          'VTK read error: ' + e.message + '</div>';
        return;
      }
      const polyData = reader.getOutputData(0);
      const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
      mapper.setInputData(polyData);
      const actor = vtk.Rendering.Core.vtkActor.newInstance();
      actor.setMapper(mapper);
      renderer.addActor(actor);
      renderer.resetCamera();
      renderWindow.render();
      // remember state for the toolbar
      this.vtkState = { fsContainer, renderer, renderWindow, mapper, actor };
      // populate range
      const arr = polyData.getPointData().getScalars();
      if (arr) {
        const r = arr.getRange();
        this.vtkRange = [r[0], r[1]];
      }
    },

    resetVtkCamera() {
      if (!this.vtkState) return;
      this.vtkState.renderer.resetCamera();
      this.vtkState.renderWindow.render();
    },
    applyVtkRange() {
      if (!this.vtkState) return;
      const m = this.vtkState.mapper;
      if (m && m.setScalarRange) {
        m.setScalarRange(this.vtkRange[0], this.vtkRange[1]);
        this.vtkState.renderWindow.render();
      }
    },
    saveVtkScreenshot() {
      if (!this.vtkState) return;
      const canvas = this.vtkState.fsContainer.getOpenGLRenderWindow()
        .getCanvas();
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = (this.viz && this.viz.name) ? this.viz.name + '.png'
        : 'render.png';
      a.click();
    },

    rerunWithParams() {
      const summary = this.params
        .map(p => `${p.name} = ${p.value}`).join('; ');
      this.prompt = 'Re-run the previous script with these parameter ' +
        'changes and report the result file: ' + summary;
      this.send();
    },

    // ─── Helpers ─────────────────────────────────────────────
    bubbleClass(e) {
      const m = {
        user_msg:           'border-accent-500/30 bg-accent-500/8 text-slate-100 ml-auto',
        agent_msg:          'border-ink-700 bg-ink-800/70 text-slate-100',
        agent_chunk:        'border-ink-700 bg-ink-800/70 text-slate-100',
        tool_call_pending:  'border-amber-500/40 bg-amber-500/8 text-slate-100',
        tool_call_executing:'border-amber-500/40 bg-amber-500/12 text-slate-100',
        tool_result:        'border-ink-700 bg-ink-900/80 text-slate-300',
        subagent_spawned:   'border-violet-500/40 bg-violet-500/10 text-slate-100',
        subagent_returned:  'border-violet-500/40 bg-violet-500/8 text-slate-200',
        token_count:        'border-ink-700/50 bg-ink-900/40 text-slate-500 text-[10px] py-1.5',
        error:              'border-rose-500/40 bg-rose-500/12 text-rose-100',
        done:               'border-accent-500/30 bg-accent-500/8 text-accent-300',
        status:             'border-ink-700/50 bg-ink-900/40 text-slate-500 text-[10px] py-1.5',
      };
      return m[e.type] || 'border-ink-700 bg-ink-800/50 text-slate-200';
    },
    eventIcon(e) {
      return ({
        user_msg: '👤', agent_msg: '🤖', agent_chunk: '🤖',
        tool_call_pending: '⚙', tool_call_executing: '⚙',
        tool_result: '✓',
        subagent_spawned: '👁', subagent_returned: '✓',
        token_count: '∑', error: '⚠', done: '●', status: '·',
      })[e.type] || '·';
    },
    eventLabel(e) {
      const m = {
        user_msg: 'You', agent_msg: 'Agent', agent_chunk: 'Agent',
        tool_call_pending: 'Tool call (pending)',
        tool_call_executing: 'Tool call',
        tool_result: 'Tool result',
        subagent_spawned: 'Sub-agent',
        subagent_returned: 'Sub-agent return',
        token_count: 'Tokens',
        error: 'Error', done: 'Done', status: 'Status',
      };
      return m[e.type] || e.type;
    },
    eventClass(e) { return this.bubbleClass(e); },
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
