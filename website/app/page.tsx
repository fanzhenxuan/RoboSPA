"use client";

import { useState } from "react";

const basePath = "/RoboSPA";

const stats = [
  ["527K", "trajectories"],
  ["997h", "interaction video"],
  ["108M", "timesteps"],
  ["56", "base tasks"],
  ["280", "task variants"],
  ["5", "embodiments"],
];

const challenges = [
  {
    index: "01",
    title: "Fine-grained grounding",
    text: "Select the intended target among visually similar candidates using geometry, distance, viewpoint, relations, and canonical indices.",
  },
  {
    index: "02",
    title: "Extended execution",
    text: "Complete multi-step procedures with repetition, ordering constraints, heterogeneous actions, and task-relevant memory.",
  },
  {
    index: "03",
    title: "Scalable complexity",
    text: "Measure how behavior degrades as candidate counts, action horizons, distractors, and environment diversity increase.",
  },
];

const categories = {
  spatial: [
    ["GAC", "Geometric Attribute Cognition", "Identify objects by subtle geometric or shape-related attributes beyond category recognition."],
    ["SDE", "Spatial Distance Estimation", "Compare distances to a reference object and reason about relative proximity."],
    ["CPI", "Canonical Position Indexing", "Ground row-column indices under different counting and scanning directions."],
    ["RRR", "Referential Relational Reasoning", "Locate targets through directional relations to reference objects."],
    ["CVR", "Cross-View Reasoning", "Transform spatial references across non-egocentric viewpoints."],
  ],
  procedural: [
    ["RPF", "Repetitive Procedure Following", "Repeat a specified operation while tracking progress and stopping correctly."],
    ["OFE", "Order-Free Execution", "Cover every subgoal in a flexible order without omission or unnecessary repetition."],
    ["OCE", "Order-Constrained Execution", "Execute subgoals in the required temporal order."],
    ["CAC", "Composite Action Coordination", "Coordinate heterogeneous manipulation skills across multi-stage procedures."],
    ["MIP", "Memory-Intensive Planning", "Retain and use information after visual cues become unavailable."],
  ],
};

const models = [
  { name: "RDT", l1: 16.8, l5: 6.9, color: "#82a966" },
  { name: "GO-1", l1: 25.1, l5: 8.8, color: "#dfab3e" },
  { name: "π0.5", l1: 55.2, l5: 22.3, color: "#416cad" },
  { name: "X-VLA", l1: 50.4, l5: 19.9, color: "#785db7" },
];

const citation = `@inproceedings{robospa2026,
  title     = {RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?},
  author    = {Zhenxuan Fan and Bo Zhang and Yutong Lin and Yuqian Yuan and Juekai Lin and Liang Liang and Zhuoyi Huang and Wenqiao Zhang and Juncheng Li and Siliang Tang and Jun Xiao and Yueting Zhuang},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}`;

export default function Home() {
  const [activeTaxonomy, setActiveTaxonomy] = useState<"spatial" | "procedural">("spatial");
  const [copied, setCopied] = useState(false);

  const copyCitation = async () => {
    await navigator.clipboard.writeText(citation);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <main>
      <nav className="topbar" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="RoboSPA home">
          <span className="brand-mark" aria-hidden="true">R</span>
          <span>RoboSPA</span>
        </a>
        <div className="nav-links">
          <a href="#abstract">Abstract</a>
          <a href="#benchmark">Benchmark</a>
          <a href="#results">Results</a>
          <a href="#citation">Citation</a>
        </div>
        <a className="nav-paper" href={`${basePath}/robospa-paper.pdf`} target="_blank" rel="noreferrer">
          Read paper <span aria-hidden="true">↗</span>
        </a>
      </nav>

      <section className="hero" id="top">
        <div className="hero-glow hero-glow-one" aria-hidden="true" />
        <div className="hero-glow hero-glow-two" aria-hidden="true" />
        <div className="hero-copy">
          <div className="eyebrow venue-badge"><span /> EMNLP 2026</div>
          <h1>
            <span className="title-line">RoboSPA: Can VLA models go beyond</span>
            <span className="title-line"><em>simple scenes</em> and <em>short-horizon tasks?</em></span>
          </h1>
          <div className="author-block" aria-label="Authors and affiliations">
            <div className="author-list">
              <span>Zhenxuan Fan<sup>1</sup></span>
              <span>Bo Zhang<sup>2</sup></span>
              <span>Yutong Lin<sup>1</sup></span>
              <span>Yuqian Yuan<sup>1</sup></span>
              <span>Juekai Lin<sup>1</sup></span>
              <span>Liang Liang<sup>1</sup></span>
              <span>Zhuoyi Huang<sup>3</sup></span>
              <span>Wenqiao Zhang<sup>1,*</sup></span>
              <span>Juncheng Li<sup>1,*</sup></span>
              <span>Siliang Tang<sup>1</sup></span>
              <span>Jun Xiao<sup>1</sup></span>
              <span>Yueting Zhuang<sup>1</sup></span>
            </div>
            <div className="author-affiliations">
              <span><sup>1</sup>Zhejiang University</span>
              <span><sup>2</sup>University of Electronic Science and Technology of China</span>
              <span><sup>3</sup>South China Normal University</span>
            </div>
            <div className="author-details">
              <span className="corresponding">* Corresponding authors</span>
              <a href="mailto:zxfan@zju.edu.cn">zxfan@zju.edu.cn</a>
              <a href="mailto:wenqiaozhang@zju.edu.cn">wenqiaozhang@zju.edu.cn</a>
            </div>
          </div>
          <p className="hero-lead">
            RoboSPA is a large-scale robotic manipulation benchmark for diagnosing
            fine-grained spatial reasoning and long-horizon procedural planning.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href={`${basePath}/robospa-paper.pdf`} target="_blank" rel="noreferrer">
              Paper <span aria-hidden="true">↗</span>
            </a>
            <a className="button button-secondary" href="https://github.com/fanzhenxuan/RoboSPA" target="_blank" rel="noreferrer">
              Code · GitHub <span aria-hidden="true">↗</span>
            </a>
            <a className="button button-secondary" href="https://huggingface.co/datasets/zxfan/RoboSPA" target="_blank" rel="noreferrer">
              Data · Hugging Face <span aria-hidden="true">↗</span>
            </a>
          </div>
        </div>

        <div className="hero-visual" aria-label="RoboSPA benchmark overview">
          <div className="visual-label"><span>Two capability dimensions</span><b>10 categories</b></div>
          <img src={`${basePath}/paper-assets/overview-final.png`} alt="Overview of the RoboSPA benchmark, showing ten task categories across spatial reasoning and long-horizon planning" />
        </div>
      </section>

      <section className="stat-band" aria-label="RoboSPA statistics">
        {stats.map(([value, label]) => (
          <div className="stat" key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <section className="section abstract-section" id="abstract">
        <div className="section-kicker abstract-kicker">01 / Abstract</div>
        <h2>Abstract</h2>
        <div className="abstract-copy">
          <p>
            Vision-Language-Action (VLA) models have shown promising progress in language-conditioned robotic manipulation. However, existing datasets and benchmarks mainly evaluate task completion under predefined settings, offering limited insight into model reasoning under increasing spatial and procedural complexity. We introduce RoboSPA (Robot Spatial-Procedural Assessment), a large-scale robotic manipulation dataset and benchmark for diagnosing embodied reasoning in VLA models. RoboSPA focuses on two core dimensions, Fine-Grained Spatial Reasoning and Long-Horizon Procedural Planning, covering 10 task categories and 56 base tasks. Each task is instantiated across five difficulty levels, yielding 280 variants with increasing spatial ambiguity and procedural complexity. We collect 527K trajectories across multiple embodiments and diverse scenes. Beyond binary success rate, RoboSPA introduces diagnostic metrics for more detailed evaluation. Experiments on representative VLA models show that current systems still struggle with complex spatial relations, precise low-level execution, and memory-intensive planning. These results establish RoboSPA as a challenging diagnostic benchmark for developing more capable, reliable, and generalizable embodied agents. Our data and code are available at <a href="https://github.com/fanzhenxuan/RoboSPA" target="_blank" rel="noreferrer">github.com/fanzhenxuan/RoboSPA</a>.
          </p>
        </div>
      </section>

      <section className="section intro-section" id="overview">
        <div className="section-kicker">02 / Overview</div>
        <div className="section-heading split-heading">
          <h2>A controlled stress test for embodied reasoning.</h2>
          <p>
            Existing evaluations often stop at final success in clean, short tasks.
            RoboSPA scales ambiguity, action horizon, and scene diversity while
            preserving task semantics—making model limitations measurable.
          </p>
        </div>
        <div className="challenge-grid">
          {challenges.map((challenge) => (
            <article className="challenge-card" key={challenge.index}>
              <span>{challenge.index}</span>
              <h3>{challenge.title}</h3>
              <p>{challenge.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="taxonomy-wrap" id="benchmark">
        <div className="section taxonomy-section">
          <div className="section-kicker light-kicker">03 / Capability taxonomy</div>
          <div className="section-heading taxonomy-heading">
            <h2>Two dimensions.<br />Ten precise capabilities.</h2>
            <p>Each category isolates a distinct reasoning bottleneck instead of treating manipulation tasks as undifferentiated successes or failures.</p>
          </div>
          <div className="taxonomy-tabs" role="tablist" aria-label="Capability dimension">
            <button
              className={activeTaxonomy === "spatial" ? "active" : ""}
              onClick={() => setActiveTaxonomy("spatial")}
              role="tab"
              aria-selected={activeTaxonomy === "spatial"}
            >
              <span>01</span> Fine-Grained Spatial Reasoning
            </button>
            <button
              className={activeTaxonomy === "procedural" ? "active" : ""}
              onClick={() => setActiveTaxonomy("procedural")}
              role="tab"
              aria-selected={activeTaxonomy === "procedural"}
            >
              <span>02</span> Long-Horizon Procedural Planning
            </button>
          </div>
          <div className="category-grid" role="tabpanel">
            {categories[activeTaxonomy].map(([abbr, title, text], index) => (
              <article className="category-card" key={abbr}>
                <div className="category-top"><b>{String(index + 1).padStart(2, "0")}</b><span>{abbr}</span></div>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section difficulty-section">
        <div className="section-kicker">04 / Hierarchical difficulty</div>
        <div className="section-heading split-heading">
          <h2>Complexity grows in controlled steps.</h2>
          <p>Spatial tasks add plausible candidates. Procedural tasks extend the sequence. Five aligned levels reveal when capability begins to break.</p>
        </div>
        <div className="design-frame">
          <img src={`${basePath}/paper-assets/design-final.png`} alt="RoboSPA capability taxonomy, five-level task design, and randomized scene examples" />
        </div>
        <div className="difficulty-scale" aria-label="Five difficulty levels">
          {[1, 2, 3, 4, 5].map((level) => (
            <div className={`level level-${level}`} key={level}>
              <span>L{level}</span>
              <div className="level-dots">{Array.from({ length: level + 1 }).map((_, index) => <i key={index} />)}</div>
              <small>{level === 1 ? "Foundation" : level === 5 ? "Stress test" : "Increasing load"}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="scale-wrap">
        <div className="section scale-section">
          <div className="scale-copy">
            <div className="section-kicker">05 / Scale &amp; diversity</div>
            <h2>Built across robots, scenes, and task difficulty.</h2>
            <p>RoboSPA combines clean and domain-randomized scenes across Aloha-AgileX, ARX-X5, Piper, Franka, and UR5.</p>
            <ul>
              <li><strong>463K</strong><span>randomized-scene trajectories</span></li>
              <li><strong>5×</strong><span>robotic embodiments</span></li>
              <li><strong>L1→L5</strong><span>more objects or longer horizons</span></li>
            </ul>
          </div>
          <div className="statistics-frame">
            <img src={`${basePath}/paper-assets/statistics-final.png`} alt="RoboSPA embodiment distribution, scene distribution, object counts, and trajectory lengths" />
          </div>
        </div>
      </section>

      <section className="section results-section" id="results">
        <div className="section-kicker">06 / Results</div>
        <div className="section-heading results-heading">
          <h2>Stronger models still fail as complexity rises.</h2>
          <div className="result-callout"><strong>&lt;25%</strong><span>overall success for every evaluated model at L5</span></div>
        </div>
        <div className="results-panel">
          <div className="panel-head">
            <div><span>Overall benchmark success rate</span><b>L1 versus L5</b></div>
            <div className="legend"><span><i className="legend-l1" /> L1</span><span><i className="legend-l5" /> L5</span></div>
          </div>
          <div className="model-rows">
            {models.map((model) => (
              <div className="model-row" key={model.name}>
                <strong>{model.name}</strong>
                <div className="bars">
                  <div className="bar-line"><div className="bar-fill l1-bar" style={{ width: `${model.l1}%`, background: model.color }} /><span>{model.l1}%</span></div>
                  <div className="bar-line"><div className="bar-fill l5-bar" style={{ width: `${model.l5}%`, background: model.color }} /><span>{model.l5}%</span></div>
                </div>
                <small>−{(model.l1 - model.l5).toFixed(1)} pts</small>
              </div>
            ))}
          </div>
          <div className="axis"><span>0</span><span>20</span><span>40</span><span>60%</span></div>
        </div>
        <div className="finding-grid">
          <article><span>Spatial</span><h3>Target selection stays close to chance.</h3><p>Low ONTA exposes unreliable grounding when identification depends on subtle relations instead of object category.</p></article>
          <article><span>Procedural</span><h3>Partial progress rarely becomes success.</h3><p>Progress Score exceeds final Success Rate, showing that models start procedures but fail to complete later stages reliably.</p></article>
          <article><span>Memory</span><h3>The hardest memory tasks collapse.</h3><p>All baselines fail on the most difficult memory-intensive planning settings.</p></article>
        </div>
      </section>

      <section className="diagnostic-wrap">
        <div className="section diagnostic-section">
          <div className="diagnostic-head">
            <div>
              <div className="section-kicker light-kicker">07 / Diagnostic analysis</div>
              <h2>Not just whether models fail—how they fail.</h2>
            </div>
            <p>Step-level evaluation separates grounding, execution, ordering, repetition, and memory failures.</p>
          </div>
          <div className="failure-frame"><img src={`${basePath}/paper-assets/failures-final.png`} alt="Representative grounding, manipulation, ordering, repetition, and memory failures in RoboSPA" /></div>
          <div className="error-list">
            {["Target execution", "Target grounding", "Manipulation", "Memory", "Temporal ordering", "Redundant repetition"].map((error, index) => (
              <span key={error}><b>{String(index + 1).padStart(2, "0")}</b>{error}</span>
            ))}
          </div>
        </div>
      </section>

      <section className="citation-wrap" id="citation">
        <div className="section citation-section">
          <div className="citation-copy">
            <div className="section-kicker light-kicker">08 / Resources</div>
            <h2>Use RoboSPA in your research.</h2>
            <p>Access the paper, implementation, and dataset resources for RoboSPA.</p>
            <div className="resource-buttons">
              <a href={`${basePath}/robospa-paper.pdf`} target="_blank" rel="noreferrer">Paper ↗</a>
              <a href="https://github.com/fanzhenxuan/RoboSPA" target="_blank" rel="noreferrer">Code · GitHub ↗</a>
              <a href="https://huggingface.co/datasets/zxfan/RoboSPA" target="_blank" rel="noreferrer">Data · Hugging Face ↗</a>
            </div>
          </div>
          <div className="bibtex-card">
            <div className="bibtex-head"><span>BibTeX</span><button onClick={copyCitation}>{copied ? "Copied ✓" : "Copy"}</button></div>
            <pre>{citation}</pre>
          </div>
        </div>
        <footer>
          <a className="brand footer-brand" href="#top"><span className="brand-mark">R</span><span>RoboSPA</span></a>
          <p>Robot Spatial-Procedural Assessment · 2026</p>
          <a href="#top">Back to top ↑</a>
        </footer>
      </section>
    </main>
  );
}
