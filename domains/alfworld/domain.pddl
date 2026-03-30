(define (domain alfworld)
  (:requirements :typing)
  (:types object location receptacle container surface agent)
  
  ;; ALFworld splits receptacles into types, we simplify conceptually for planner
  ;; Container = Microwave, Fridge, Cabinet, Drawer (can be opened/closed)
  ;; Surface = CounterTop, Sofa, Bed (always open)

  (:predicates
    (at ?a - agent ?l - location)
    (on ?o - object ?r - receptacle)
    (in ?o - object ?r - container)
    (holding ?a - agent ?o - object)
    (open ?r - container)
    (closed ?r - container)
    (visible ?x - object)
  )

  (:action goto_location
    :parameters (?a - agent ?from - location ?to - location)
    :precondition (and (at ?a ?from))
    :effect (and (not (at ?a ?from)) (at ?a ?to))
  )

  (:action open_receptacle
    :parameters (?a - agent ?l - location ?r - container)
    :precondition (and (at ?a ?l) (closed ?r))
    :effect (and (not (closed ?r)) (open ?r))
  )

  (:action close_receptacle
    :parameters (?a - agent ?l - location ?r - container)
    :precondition (and (at ?a ?l) (open ?r))
    :effect (and (not (open ?r)) (closed ?r))
  )

  (:action take_from_surface
    :parameters (?a - agent ?l - location ?o - object ?r - surface)
    :precondition (and (at ?a ?l) (on ?o ?r) (visible ?o))
    :effect (and (not (on ?o ?r)) (holding ?a ?o))
  )

  (:action take_from_container
    :parameters (?a - agent ?l - location ?o - object ?r - container)
    :precondition (and (at ?a ?l) (in ?o ?r) (open ?r) (visible ?o))
    :effect (and (not (in ?o ?r)) (holding ?a ?o))
  )

  (:action put_in_container
    :parameters (?a - agent ?l - location ?o - object ?r - container)
    :precondition (and (at ?a ?l) (holding ?a ?o) (open ?r))
    :effect (and (not (holding ?a ?o)) (in ?o ?r))
  )

  (:action put_on_surface
    :parameters (?a - agent ?l - location ?o - object ?r - surface)
    :precondition (and (at ?a ?l) (holding ?a ?o))
    :effect (and (not (holding ?a ?o)) (on ?o ?r))
  )

  ;; Sensing Action
  (:action look_inside
    :parameters (?a - agent ?l - location ?r - container ?o - object)
    :precondition (and (at ?a ?l) (open ?r) (not (visible ?o)))
    :effect (visible ?o)
  )
)
