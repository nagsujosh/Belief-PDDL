(define (domain alfworld)
  (:predicates
    (agent_entity ?a)
    (at ?a ?r)
    (on ?o ?r)
    (in ?o ?r)
    (holding ?a ?o)
    (open ?r)
    (closed ?r)
    (visible ?x)
  )

  (:action goto_location
    :parameters (?a ?to)
    :precondition (and (agent_entity ?a))
    :effect (and (at ?a ?to))
  )

  (:action open_receptacle
    :parameters (?a ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (closed ?r))
    :effect (and (not (closed ?r)) (open ?r))
  )

  (:action close_receptacle
    :parameters (?a ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (open ?r))
    :effect (and (not (open ?r)) (closed ?r))
  )

  (:action take_from_surface
    :parameters (?a ?o ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (on ?o ?r) (visible ?o))
    :effect (and (not (on ?o ?r)) (holding ?a ?o))
  )

  (:action take_from_container
    :parameters (?a ?o ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (in ?o ?r) (open ?r) (visible ?o))
    :effect (and (not (in ?o ?r)) (holding ?a ?o))
  )

  (:action put_in_container
    :parameters (?a ?o ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (holding ?a ?o) (open ?r))
    :effect (and (not (holding ?a ?o)) (in ?o ?r))
  )

  (:action put_on_surface
    :parameters (?a ?o ?r)
    :precondition (and (agent_entity ?a) (at ?a ?r) (holding ?a ?o))
    :effect (and (not (holding ?a ?o)) (on ?o ?r))
  )

  (:action look_inside
    :parameters (?a ?r ?o)
    :precondition (and (agent_entity ?a) (at ?a ?r) (open ?r))
    :effect (visible ?o)
  )
)
