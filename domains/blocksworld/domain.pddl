(define (domain blocksworld)
  (:requirements :typing)
  (:types block)

  (:predicates
    (on_table ?x - block)
    (on ?x - block ?y - block)
    (clear ?x - block)
    (arm_empty)
    (holding ?x - block)
    ;; To interface with top-k worlds, all sensing predicates like visible represent 
    ;; the "current state". But for pyperplan planner to work seamlessly during 
    ;; rollout, we might not strictly need them here or we treat them as fluents 
    ;; for sensing actions. Let's keep the standard PDDL logic for physical actions.
    (visible ?x - block)
  )

  (:action pickup
    :parameters (?x - block)
    :precondition (and (clear ?x) (on_table ?x) (arm_empty) (visible ?x))
    :effect (and 
      (not (on_table ?x)) 
      (not (clear ?x)) 
      (not (arm_empty)) 
      (holding ?x))
  )

  (:action putdown
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and 
      (not (holding ?x)) 
      (arm_empty) 
      (clear ?x) 
      (on_table ?x))
  )

  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (clear ?x) (on ?x ?y) (arm_empty) (visible ?x))
    :effect (and 
      (not (on ?x ?y)) 
      (not (clear ?x)) 
      (not (arm_empty)) 
      (holding ?x) 
      (clear ?y))
  )

  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y) (visible ?y))
    :effect (and 
      (not (holding ?x)) 
      (not (clear ?y)) 
      (arm_empty) 
      (on ?x ?y) 
      (clear ?x))
  )

  ;; Sensing actions are handled entirely by SampleBeliefPlanner heuristics 
  ;; and do not exist in the deterministic PDDL domain to avoid Parser confusion.
)
