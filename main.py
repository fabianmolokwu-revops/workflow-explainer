# main.py
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

# ============================================================================
# DATA MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    workflow: Dict[str, Any]
    question: str = "what does it do?"

class AskRequest(BaseModel):
    workflow: Dict[str, Any]
    question: str

class FeedbackRequest(BaseModel):
    feedback: str
    email: Optional[str] = None
    timestamp: str

class TeachNodeRequest(BaseModel):
    node_id: str
    node_name: str
    app_name: str
    purpose: str
    workflow_id: str

# ============================================================================
# SIMPLE RELIABLE PARSER (Works every time)
# ============================================================================

class WorkflowParser:
    """Simple, reliable parser that doesn't overcomplicate"""
    
    def get_app_from_node(self, node: Dict) -> str:
        """Extract app name from node type"""
        node_type = node.get("type", "").lower()
        
        # Common n8n nodes
        if "webhook" in node_type:
            return "Webhook"
        if "airtable" in node_type:
            return "Airtable"
        if "slack" in node_type:
            return "Slack"
        if "google" in node_type or "sheets" in node_type:
            return "Google Sheets"
        if "whatsapp" in node_type:
            return "WhatsApp"
        if "hubspot" in node_type:
            return "HubSpot"
        if "openai" in node_type or "gpt" in node_type:
            return "OpenAI"
        if "docusign" in node_type:
            return "DocuSign"
        if "calendly" in node_type:
            return "Calendly"
        if "notion" in node_type:
            return "Notion"
        if "salesforce" in node_type:
            return "Salesforce"
        if "stripe" in node_type:
            return "Stripe"
        if "http" in node_type:
            return "HTTP Request"
        if "if" in node_type or "filter" in node_type:
            return "Condition"
        if "code" in node_type:
            return "Code Node"
        if "wait" in node_type:
            return "Wait"
        if "schedule" in node_type or "trigger" in node_type:
            return "Schedule"
        if "error" in node_type:
            return "Error Handler"
        if "langchain" in node_type or "agent" in node_type:
            return "AI Agent"
        
        # Fallback: extract from node type
        parts = node_type.split(".")
        if len(parts) > 1:
            return parts[-1].replace("Trigger", "").replace("Request", "").strip()
        
        return "Unknown"
    
    def parse(self, workflow_json: Dict) -> Dict:
        """Parse workflow into trigger, actions, dependencies"""
        nodes = workflow_json.get("nodes", [])
        
        if not nodes:
            return {"trigger": None, "actions": [], "dependencies": [], "trigger_index": -1}
        
        # Find trigger (webhook, schedule, or first node)
        trigger_index = -1
        trigger = None
        
        for i, node in enumerate(nodes):
            node_type = node.get("type", "").lower()
            if "webhook" in node_type or "schedule" in node_type or "trigger" in node_type:
                trigger_index = i
                trigger = node
                break
        
        # If no webhook/schedule, first node is trigger
        if trigger_index == -1 and nodes:
            trigger_index = 0
            trigger = nodes[0]
        
        # Actions = all nodes except trigger
        actions = []
        for i, node in enumerate(nodes):
            if i != trigger_index:
                actions.append(node)
        
        # Extract dependencies (unique apps)
        dependencies = set()
        for node in nodes:
            app = self.get_app_from_node(node)
            if app != "Unknown":
                dependencies.add(app)
        
        return {
            "trigger": trigger,
            "actions": actions,
            "dependencies": list(dependencies),
            "trigger_index": trigger_index,
            "total_nodes": len(nodes)
        }


# ============================================================================
# RISK ANALYZER
# ============================================================================

class RiskAnalyzer:
    """Simple risk detection based on dependencies"""
    
    def analyze(self, dependencies: List[str]) -> List[str]:
        risks = []
        
        risk_map = {
            "Webhook": "Webhook endpoint could fail → no data enters system",
            "Airtable": "Airtable API rate limits → record operations may fail",
            "Slack": "Slack outage → team misses notifications",
            "Google Sheets": "Google Sheets API quota → logging could stop",
            "WhatsApp": "WhatsApp API limits → messages may be delayed",
            "HubSpot": "HubSpot API failure → CRM operations stop",
            "OpenAI": "OpenAI API costs → unexpected charges possible",
            "Stripe": "Stripe payment failure → transactions fail",
            "HTTP Request": "External API changes → workflow may break",
            "AI Agent": "AI model latency or errors → responses delayed"
        }
        
        for dep in dependencies:
            if dep in risk_map:
                risks.append(risk_map[dep])
        
        if not risks:
            risks.append("No specific risks detected")
        
        return risks


# ============================================================================
# HEALTH SCORE CALCULATOR
# ============================================================================

class HealthScoreCalculator:
    """Calculate health score from documentation and complexity"""
    
    def calculate(self, workflow_json: Dict, dependencies: List[str], action_count: int) -> Dict:
        score = 100
        breakdown = []
        
        # Documentation (max 40 points)
        if workflow_json.get("business_purpose"):
            breakdown.append("✓ Business purpose documented (+15)")
        else:
            score -= 15
            breakdown.append("✗ Missing business purpose (-15)")
        
        if workflow_json.get("owner"):
            breakdown.append("✓ Owner assigned (+10)")
        else:
            score -= 10
            breakdown.append("✗ Missing owner (-10)")
        
        if workflow_json.get("business_impact"):
            breakdown.append("✓ Business impact stated (+15)")
        else:
            score -= 15
            breakdown.append("✗ Missing business impact (-15)")
        
        # Recovery (max 20 points)
        if workflow_json.get("failure_procedure") or workflow_json.get("on_error"):
            breakdown.append("✓ Failure recovery documented (+20)")
        else:
            score -= 20
            breakdown.append("✗ Missing failure recovery (-20)")
        
        # Review (max 15 points)
        if workflow_json.get("last_reviewed"):
            breakdown.append("✓ Recently reviewed (+15)")
        else:
            score -= 15
            breakdown.append("✗ No last review date (-15)")
        
        # Complexity (max 10 points - more nodes = lower score)
        if action_count <= 5:
            breakdown.append("✓ Low complexity (+10)")
        elif action_count <= 10:
            breakdown.append("✓ Medium complexity (+5)")
            score -= 5
        else:
            breakdown.append("✗ High complexity (-10)")
            score -= 10
        
        # Dependency risk (max 10 points)
        if len(dependencies) <= 3:
            breakdown.append("✓ Few dependencies (+10)")
        elif len(dependencies) <= 6:
            breakdown.append("✓ Moderate dependencies (+5)")
            score -= 5
        else:
            breakdown.append("✗ Many dependencies (-10)")
            score -= 10
        
        # Cap at 0
        score = max(0, min(100, score))
        
        # Determine status
        if score >= 80:
            status = "Good"
        elif score >= 50:
            status = "Needs Improvement"
        else:
            status = "Poor"
        
        return {
            "score": score,
            "status": status,
            "breakdown": breakdown
        }


# ============================================================================
# PURPOSE INFERENCE (Simple - no hardcoded confidence)
# ============================================================================

class PurposeInference:
    """Infer purpose from workflow data when not documented"""
    
    def __init__(self, ai_client=None):
        self.client = ai_client
    
    def infer(self, workflow_json: Dict, trigger: Dict, actions: List[Dict]) -> Dict:
        """Infer purpose with evidence-based confidence"""
        
        workflow_name = workflow_json.get("name", "")
        nodes = [trigger] + actions if trigger else actions
        
        # Collect evidence
        evidence = []
        confidence = 0
        
        # Evidence 1: Workflow name (max 20%)
        if workflow_name and workflow_name != "Unnamed":
            evidence.append(f"Workflow name: {workflow_name}")
            confidence += 15
        else:
            confidence += 0
        
        # Evidence 2: Node names (max 30%)
        node_names = [n.get("name", "") for n in nodes if n.get("name")]
        business_keywords = ["lead", "customer", "patient", "order", "payment", "invoice", 
                            "notification", "alert", "report", "sync", "backup", "export"]
        
        keyword_matches = 0
        for name in node_names:
            name_lower = name.lower()
            for kw in business_keywords:
                if kw in name_lower:
                    keyword_matches += 1
                    break
        
        if keyword_matches >= 3:
            confidence += 30
            evidence.append(f"Node names suggest business process ({keyword_matches} keywords)")
        elif keyword_matches >= 1:
            confidence += 15
            evidence.append(f"Node names hint at purpose ({keyword_matches} keywords)")
        
        # Evidence 3: Trigger type (max 20%)
        if trigger:
            trigger_type = trigger.get("type", "").lower()
            if "webhook" in trigger_type:
                confidence += 20
                evidence.append("Trigger: Webhook (receives external data)")
            elif "schedule" in trigger_type:
                confidence += 15
                evidence.append("Trigger: Schedule (runs on timer)")
            elif "manual" in trigger_type:
                confidence += 5
                evidence.append("Trigger: Manual (human-initiated)")
        
        # Evidence 4: Dependencies (max 20%)
        apps = set()
        for node in nodes:
            node_type = node.get("type", "").lower()
            if "airtable" in node_type:
                apps.add("Airtable")
            elif "slack" in node_type:
                apps.add("Slack")
            elif "google" in node_type:
                apps.add("Google")
            elif "hubspot" in node_type:
                apps.add("HubSpot")
        
        if len(apps) >= 2:
            confidence += 20
            evidence.append(f"Integrates {len(apps)} systems: {', '.join(list(apps)[:3])}")
        elif len(apps) == 1:
            confidence += 10
            evidence.append(f"Integrates with {list(apps)[0]}")
        
        # Evidence 5: Action count (max 10%)
        if len(actions) >= 3:
            confidence += 10
            evidence.append(f"Multi-step workflow ({len(actions)} actions)")
        
        # Cap confidence at 95
        confidence = min(95, confidence)
        
        # Generate purpose statement using AI if available and confidence is decent
        purpose_statement = None
        if self.client and confidence >= 30:
            try:
                context = f"Workflow: {workflow_name}\n"
                context += f"Trigger: {trigger.get('name', 'Unknown') if trigger else 'Unknown'}\n"
                context += f"Actions: {', '.join([a.get('name', 'Unknown') for a in actions[:5]])}\n"
                context += f"Apps: {', '.join(apps) if apps else 'Unknown'}"
                
                prompt = f"""Based on this workflow, what is its business purpose?

{context}

Write ONE sentence describing what this workflow does. Be specific but concise.

Purpose:"""
                
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                purpose_statement = response.choices[0].message.content.strip()
            except:
                pass
        
        # If no AI or AI failed, use template
        if not purpose_statement:
            if "lead" in workflow_name.lower():
                purpose_statement = f"Processes and manages leads through {', '.join(apps) if apps else 'multiple systems'}"
            elif "notification" in workflow_name.lower():
                purpose_statement = "Sends automated notifications based on triggers"
            elif "sync" in workflow_name.lower():
                purpose_statement = "Synchronizes data between connected systems"
            else:
                purpose_statement = f"Automates workflow from {trigger.get('name', 'trigger') if trigger else 'start'} through {len(actions)} steps"
        
        return {
            "purpose": purpose_statement,
            "confidence": confidence,
            "evidence": evidence,
            "is_inferred": not workflow_json.get("business_purpose")
        }


# ============================================================================
# DELETE IMPACT ANALYZER
# ============================================================================

class DeleteImpactAnalyzer:
    """Analyze what breaks if workflow is deleted"""
    
    def __init__(self, ai_client=None):
        self.client = ai_client
    
    def analyze(self, workflow_json: Dict, purpose: str, dependencies: List[str], action_count: int) -> Dict:
        """Return delete safety assessment"""
        
        confidence = 0
        factors = []
        what_breaks = []
        
        # Factor 1: Has business impact documented (max 30%)
        business_impact = workflow_json.get("business_impact", "").lower()
        if "critical" in business_impact or "high" in business_impact:
            confidence += 30
            factors.append("Documented as CRITICAL/HIGH impact (+30)")
        elif business_impact:
            confidence += 15
            factors.append("Business impact documented (+15)")
        else:
            factors.append("No business impact documented (0)")
        
        # Factor 2: Has owner (max 10%)
        if workflow_json.get("owner"):
            confidence += 10
            factors.append("Has assigned owner (+10)")
        
        # Factor 3: Dependency count (max 25%)
        if len(dependencies) > 5:
            confidence += 25
            factors.append(f"Many dependencies ({len(dependencies)}) - high risk (+25)")
        elif len(dependencies) > 2:
            confidence += 15
            factors.append(f"Moderate dependencies ({len(dependencies)}) (+15)")
        elif len(dependencies) > 0:
            confidence += 5
            factors.append(f"Few dependencies ({len(dependencies)}) (+5)")
        
        # Factor 4: Action complexity (max 15%)
        if action_count > 10:
            confidence += 15
            factors.append(f"Complex workflow ({action_count} actions) (+15)")
        elif action_count > 5:
            confidence += 8
            factors.append(f"Multi-step workflow ({action_count} actions) (+8)")
        
        # Factor 5: Purpose clarity (max 20%)
        if purpose and len(purpose) > 20:
            confidence += 20
            factors.append("Clear business purpose identified (+20)")
        elif purpose:
            confidence += 10
            factors.append("Has business purpose (+10)")
        
        # Cap confidence
        confidence = min(100, confidence)
        
        # Determine recommendation
        if confidence >= 70:
            can_delete = False
            recommendation = "DO NOT DELETE - Critical workflow"
            what_breaks = ["Core business process would stop", "Multiple dependencies would fail", "No documented replacement"]
        elif confidence >= 40:
            can_delete = False
            recommendation = "PROBABLY NOT - Investigate before deleting"
            what_breaks = ["May impact dependent processes", "Business impact unclear"]
        elif confidence >= 15:
            can_delete = True
            recommendation = "MAY BE SAFE - But verify with team"
            what_breaks = ["Unknown - limited documentation"]
        else:
            can_delete = True
            recommendation = "INSUFFICIENT DATA - Manual review required"
            what_breaks = ["Unable to determine impact"]
        
        return {
            "can_delete": can_delete,
            "confidence": confidence,
            "confidence_factors": factors,
            "what_breaks": what_breaks[:3],
            "recommendation": recommendation
        }


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Workflow Explainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
ai_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

parser = WorkflowParser()
risk_analyzer = RiskAnalyzer()
health_calculator = HealthScoreCalculator()
purpose_inference = PurposeInference(ai_client)
delete_analyzer = DeleteImpactAnalyzer(ai_client)

# Knowledge base for learning
KNOWLEDGE_FILE = "knowledge.json"

def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, 'r') as f:
            return json.load(f)
    return {"node_patterns": {}, "workflow_memory": {}}

def save_knowledge(knowledge):
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump(knowledge, f, indent=2)

knowledge = load_knowledge()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Workflow Explainer API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "ai_available": ai_client is not None}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Main analysis endpoint - simple, reliable"""
    try:
        workflow_json = request.workflow
        workflow_name = workflow_json.get("name", "Unnamed")
        
        # Parse workflow
        parsed = parser.parse(workflow_json)
        trigger = parsed["trigger"]
        actions = parsed["actions"]
        dependencies = parsed["dependencies"]
        
        # Extract basic info
        trigger_name = trigger.get("name", "Unknown") if trigger else "None"
        action_names = [a.get("name", "Unknown") for a in actions]
        
        # Get documented info
        business_purpose = workflow_json.get("business_purpose", "")
        owner = workflow_json.get("owner", "")
        business_impact = workflow_json.get("business_impact", "")
        created_at = workflow_json.get("created_at", workflow_json.get("created", ""))
        
        # Infer purpose if not documented
        inferred = None
        if not business_purpose:
            inferred = purpose_inference.infer(workflow_json, trigger, actions)
            business_purpose = inferred.get("purpose", "")
        
        # Generate summary (use documented purpose first)
        if workflow_json.get("business_purpose"):
            summary = workflow_json.get("business_purpose")
        elif inferred and inferred.get("purpose"):
            summary = inferred.get("purpose")
        else:
            summary = f"{trigger_name} → {' → '.join(action_names[:3])}" if action_names else f"{trigger_name} workflow"
        
        # Analyze risks
        risks = risk_analyzer.analyze(dependencies)
        
        # Calculate health score
        missing_docs = []
        if not workflow_json.get("business_purpose") and not inferred:
            missing_docs.append("No business purpose documented")
        if not owner:
            missing_docs.append("No owner assigned")
        if not workflow_json.get("failure_procedure"):
            missing_docs.append("No failure recovery steps")
        if not workflow_json.get("last_reviewed"):
            missing_docs.append("No last review date")
        
        health = health_calculator.calculate(workflow_json, dependencies, len(actions))
        
        return {
            "name": workflow_name,
            "summary": summary,
            "trigger": trigger_name,
            "actions": action_names[:10],
            "dependencies": dependencies,
            "risks": risks[:5],
            "health_score": health["score"],
            "health_status": health["status"],
            "health_breakdown": health["breakdown"],
            "missing_docs": missing_docs,
            "business_purpose": business_purpose,
            "inferred_purpose": inferred.get("purpose") if inferred and not workflow_json.get("business_purpose") else None,
            "purpose_confidence": inferred.get("confidence") if inferred else 0,
            "owner": owner,
            "created_at": created_at,
            "business_impact": business_impact,
            "stats": {
                "total_nodes": parsed["total_nodes"],
                "action_count": len(actions)
            }
        }
    except Exception as e:
        print(f"Error in /analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impact")
async def delete_impact(request: AnalyzeRequest):
    """Delete safety analysis"""
    try:
        workflow_json = request.workflow
        
        parsed = parser.parse(workflow_json)
        trigger = parsed["trigger"]
        actions = parsed["actions"]
        dependencies = parsed["dependencies"]
        
        # Get or infer purpose
        business_purpose = workflow_json.get("business_purpose", "")
        if not business_purpose:
            inferred = purpose_inference.infer(workflow_json, trigger, actions)
            purpose = inferred.get("purpose", "")
            purpose_confidence = inferred.get("confidence", 0)
        else:
            purpose = business_purpose
            purpose_confidence = 100
        
        # Analyze delete impact
        impact = delete_analyzer.analyze(
            workflow_json, purpose, dependencies, len(actions)
        )
        
        return impact
    except Exception as e:
        return {
            "can_delete": True,
            "confidence": 0,
            "confidence_factors": [f"Error: {str(e)[:100]}"],
            "what_breaks": ["Unable to analyze - manual review needed"],
            "recommendation": "Review workflow manually"
        }

@app.post("/ask")
async def ask(request: AskRequest):
    """Ask questions about the workflow"""
    try:
        workflow_json = request.workflow
        question = request.question
        
        if not ai_client:
            return {"answer": "AI not configured. Add GROQ_API_KEY environment variable."}
        
        # Parse workflow for context
        parsed = parser.parse(workflow_json)
        trigger = parsed["trigger"]
        actions = parsed["actions"]
        dependencies = parsed["dependencies"]
        
        trigger_name = trigger.get("name", "Unknown") if trigger else "None"
        action_names = [a.get("name", "Unknown") for a in actions[:5]]
        
        # Build context
        context = f"""Workflow Name: {workflow_json.get('name', 'Unnamed')}
Business Purpose: {workflow_json.get('business_purpose', 'Not documented')}
Owner: {workflow_json.get('owner', 'Not documented')}
Trigger: {trigger_name}
Actions: {', '.join(action_names)}
Dependencies: {', '.join(dependencies)}
"""

        prompt = f"""You are a workflow analyst. Answer the user's question based on the workflow data below.

{context}

User Question: {question}

Answer concisely (2-3 sentences). If the answer isn't in the data, say so honestly.

Answer:"""

        response = ai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        return {"answer": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"answer": f"Error: {str(e)[:100]}"}

@app.get("/ui")
async def serve_ui():
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>")

@app.post("/teach")
async def teach(request: TeachNodeRequest):
    """Learn from user feedback"""
    knowledge["node_patterns"][request.node_name] = {
        "app": request.app_name,
        "purpose": request.purpose,
        "learned_at": datetime.now().isoformat()
    }
    save_knowledge(knowledge)
    return {"status": "success", "message": f"Thanks! I'll remember that {request.node_name} is {request.app_name}."}

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Save user feedback"""
    feedback_file = "feedback.json"
    feedbacks = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedbacks = json.load(f)
    feedbacks.append(request.dict())
    with open(feedback_file, 'w') as f:
        json.dump(feedbacks, f, indent=2)
    return {"status": "success", "message": "Thank you for your feedback!"}

@app.get("/knowledge")
async def get_knowledge():
    """View learned patterns"""
    return {
        "node_patterns": knowledge.get("node_patterns", {}),
        "patterns_count": len(knowledge.get("node_patterns", {}))
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Workflow Explainer API starting on http://localhost:{port}")
    print(f"📊 Health: http://localhost:{port}/health")
    print(f"🎨 UI: http://localhost:{port}/ui")
    print(f"💥 Delete Impact: POST http://localhost:{port}/impact")
    uvicorn.run(app, host="0.0.0.0", port=port)
