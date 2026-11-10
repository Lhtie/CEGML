# Verifiable Symbolic Reasoning and Learning

## 1. Multi-domain Training for Counterexample Utilization

We propose to investigate whether counterexample utilization can be learned as a general and transferable reasoning skill through multi-domain training.

The central idea is to jointly train an LLM on a diverse collection of verifiable symbolic reasoning tasks. These domains may include:

1. regular-expression synthesis

2. propositional logic

3. linear temporal logic

4. program synthesis/repair

5. loop-invariant generation

6. ARC puzzles

7. database-query synthesis

Although their surface representations differ, they share a common iterative structure: the model produces a symbolic hypothesis, a domain-specific verifier evaluates it, the verifier returns a counterexample, and the model revises its hypothesis. We hypothesize that exposure to this common structure across many domains will teach the model general strategies for interpreting counterexamples, locating the source of an error, and producing targeted revisions.

A central challenge is how to represent information consistently across domains. We may propose a unified counterexample-guided episode consisting of a task specification, a candidate hypothesis, a verification result, a counterexample, an explanation of the behavioral mismatch, and a revised hypothesis. Domain-specific objects—such as strings for regular expressions, traces for LTL, program states for invariants, database instances for SQL, or grids for ARC—will remain in their native representations. The design should preserve domain-specific meaning while making the abstract process of counterexample-guided revision consistent across tasks.

We will evaluate whether multi-domain training improves counterexample efficiency. The main comparison will be between single-domain models, and multi-domain models explicitly trained on counterexample-guided trajectories. If successful, this work would show that multi-task learning can transfer not only task knowledge, but also a general procedure for learning from formal feedback.

We can also evaluate transfer ability on held-out domains after joint multi-domain training.

## 2. Test-time Training/RL

Starting from the jointly trained multi-domain model, we can further use test-time training to adapt the model to a specific symbolic domain or problem instance. The multi-domain model provides a general prior for hypothesis generation, verifier interaction, counterexample interpretation. At test time, it can be temporarily updated using domain-specific demonstrations, verifier-generated counterexamples, and successful interaction trajectories collected from the current task. To make this process efficient and avoid catastrophic forgetting, the base model can remain frozen while only a temporary LoRA module, adapter, or soft prompt is updated. The adapted parameters are used to solve the current domain or task family and can then be discarded or reset. This framework combines the transferable counterexample-handling abilities acquired through multi-domain training with the specialized syntax, semantics, verifier behavior, and repair patterns required by an individual domain.

## 3. Tool-Using Agents for Active Counterexample-Guided Reasoning

After talking with Cody about his project, I realized that counterexamples can also be used implicitly. In his project, counterexample is used to guide coming up with intermediate lemma in order to proof the whole theorem. 

This gives a research direction that develops tool-using agents that can actively decide how to interact with symbolic verifiers. Instead of following a fixed pipeline in which every candidate is automatically checked and every returned counterexample is directly inserted into the prompt, the LLM agent will plan when and how to use the verifier. The objective is therefore not only to produce a correct symbolic representation, but also to learn an effective verification and debugging strategy.

## 4. Explore other Forms of Feedback

When we have a dedicated agent to decide how to retrieve and make use of the verifier provided feedback, it can also decide the type/form of feedback/information the verifier provides beyond merely counterexamples.


