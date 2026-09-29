"use strict";
const ui = (id) => document.getElementById(id);
let offset = 0, selected = null, requestVersion = 0, listVersion = 0;
const limit = 15;
const display = (value, suffix = "") => value === null || value === undefined ? "未知" : `${value}${suffix}`;
function element(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; }
function notice(text, error = false) { ui("notice").textContent = text; ui("notice").className = error ? "error-text" : ""; }
async function api(path) {
  const response = await fetch(path, {headers:{Authorization:`Bearer ${ui("token").value.trim()}`}, cache:"no-store"});
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `请求失败 (${response.status})`);
  return body;
}
function filters() {
  const query = new URLSearchParams({limit, offset});
  for (const [id, key] of [["status","status"],["thread","thread_id"],["task","task_id"],["node","node"],["error-type","error_type"]]) if(ui(id).value) query.set(key,ui(id).value);
  for(const key of ["start","end"]) if(ui(key).value) query.set(key,new Date(ui(key).value).getTime()/1000);
  return query;
}
async function loadList() {
  const version = ++listVersion;
  notice("正在读取轨迹…");
  try {
    const data = await api(`/api/traces?${filters()}`);
    if(version !== listVersion) return;
    ui("trace-list").replaceChildren(); ui("total").textContent = `${data.total} 条`;
    ui("previous").disabled=offset===0; ui("next").disabled=offset+limit>=data.total;
    for(const trace of data.items) {
      const button=element("button",undefined,"trace-row"); button.type="button";
      const top=element("div",undefined,"row-top"); top.append(element("span",new Date(trace.start_time*1000).toLocaleString()),element("span",trace.status,`badge ${trace.status}`));
      button.append(top,element("code",trace.trace_id),element("small",`任务 ${trace.task_id||"—"}\n${display(trace.total_latency_ms," ms")} · Token ${tokenTotal(trace)} · 工具 ${display(trace.total_tool_calls)} · 异常 ${display(trace.error_count)}`));
      button.addEventListener("click",()=>selectTrace(trace.trace_id,button));ui("trace-list").append(button);
    }
    if(!data.items.length) ui("trace-list").append(element("p","没有符合条件的轨迹","empty"));
    notice(`已加载 ${data.items.length} 条。输入输出默认仅保留脱敏摘要；“未知”表示接口未提供指标。`);
    const wanted=new URLSearchParams(location.search).get("trace_id");
    if(wanted && !selected) await selectTrace(wanted);
  } catch(error) {if(version !== listVersion) return;notice(error.message,true);ui("trace-list").replaceChildren();}
}
function tokenTotal(trace) { return trace.total_input_tokens==null||trace.total_output_tokens==null ? "未知" : trace.total_input_tokens+trace.total_output_tokens; }
function showSpan(span) {
  ui("span-meta").replaceChildren(element("strong",span.name),element("span",`${span.span_type} · ${span.status} · ${display(span.duration_ms," ms")}`,`badge ${span.status}`));
  ui("span-detail").textContent=JSON.stringify({span_id:span.span_id,parent_span_id:span.parent_span_id,input:span.input_summary,output:span.output_summary,attributes:span.attributes,error_type:span.error_type,error:span.error_message},null,2);
}
function treeNode(span) {
  const wrapper=element("details");wrapper.open=true;
  const summary=element("summary");const button=element("button",`${span.name} · ${display(span.duration_ms," ms")}`);button.type="button";
  button.addEventListener("click",(event)=>{event.preventDefault();showSpan(span);});
  summary.append(button,element("span",span.span_type,`badge ${span.status}`));wrapper.append(summary);
  for(const child of span.children) wrapper.append(treeNode(child));
  return wrapper;
}
async function selectTrace(id, button) {
  const version=++requestVersion;
  try {
    const data=await api(`/api/traces/${encodeURIComponent(id)}`);if(version!==requestVersion)return;
    selected=data;document.querySelectorAll(".trace-row.active").forEach(row=>row.classList.remove("active"));if(button)button.classList.add("active");
    const trace=data.trace;ui("selected-title").textContent=`请求 ${trace.trace_id.slice(0,12)}…`;
    ui("correlation").textContent=`会话 ${trace.thread_id||"—"} · 任务 ${trace.task_id||"—"} · 业务结果 ${trace.outcome||"未知"}`;
    ui("metrics").replaceChildren();
    for(const [label,value] of [["总耗时",display(trace.total_latency_ms," ms")],["模型调用",display(trace.total_llm_calls)],["输入 / 输出 Token",`${display(trace.total_input_tokens)} / ${display(trace.total_output_tokens)}`],["工具调用",display(trace.total_tool_calls)],["重规划",display(trace.replan_count)],["异常 Span",display(trace.error_count)]]) {
      const card=element("div",undefined,"metric");card.append(element("span",label),element("strong",value));ui("metrics").append(card);
    }
    ui("tree").replaceChildren(...data.tree.map(treeNode));ui("timeline").replaceChildren();
    const start=trace.start_time*1000;const duration=Math.max(trace.total_latency_ms||1,1);
    for(const span of data.spans) {
      const row=element("div",undefined,"timeline-row");row.tabIndex=0;row.setAttribute("role","button");
      const track=element("div",undefined,"track");const bar=element("div",undefined,`bar ${span.span_type} ${span.status}`);
      bar.style.left=`${Math.max(0,Math.min(99,(span.start_time*1000-start)/duration*100))}%`;
      bar.style.width=`${Math.min(100,(span.duration_ms||0)/duration*100)}%`;track.append(bar);
      row.append(element("span",span.name,"timeline-name"),track,element("span",display(span.duration_ms,"ms")));
      row.addEventListener("click",()=>showSpan(span));row.addEventListener("keydown",e=>{if(e.key==="Enter")showSpan(span);});ui("timeline").append(row);
    }
    ui("plans").replaceChildren();
    const events=data.events.filter(e=>/GOAL|PLAN|RECOVERY|UNCERTAIN|HANDOFF|POLICY|EVIDENCE|EVALUATOR|CORRECTIVE|RAG/.test(e.event_type));
    for(const ev of events){const row=element("div",undefined,"plan-event");row.append(element("strong",`${ev.event_type} · 计划 ${ev.plan_version??"—"}`),element("small",` · ${new Date(ev.timestamp*1000).toLocaleTimeString()}`),element("pre",JSON.stringify(ev.attributes,null,2)));ui("plans").append(row);}
    if(!events.length)ui("plans").append(element("p","该请求没有计划变更或恢复事件","empty"));
    ui("export").disabled=false;if(data.spans.length)showSpan(data.spans[0]);
    notice(trace.incomplete?"此轨迹存在中断或采集上限，部分指标未知。":"已加载完整执行树。");
  }catch(error){notice(error.message,true);}
}
ui("filters").addEventListener("submit",e=>{e.preventDefault();offset=0;loadList();});
ui("previous").addEventListener("click",()=>{offset=Math.max(0,offset-limit);loadList();});
ui("next").addEventListener("click",()=>{offset+=limit;loadList();});
ui("token").addEventListener("input",()=>{selected=null;requestVersion++;listVersion++;ui("selected-title").textContent="选择一条轨迹";ui("correlation").textContent="";ui("span-meta").replaceChildren();ui("export").disabled=true;ui("trace-list").replaceChildren();ui("tree").replaceChildren();ui("timeline").replaceChildren();ui("plans").replaceChildren();ui("metrics").replaceChildren();ui("span-detail").textContent="凭证已改变，请重新查询";});
ui("export").addEventListener("click",async()=>{if(!selected)return;try{const data=await api(`/api/traces/${selected.trace.trace_id}/export`);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:"application/json"}));const a=element("a");a.href=url;a.download=`trace-${data.trace.trace_id}.json`;a.click();URL.revokeObjectURL(url);}catch(error){notice(error.message,true);}});
