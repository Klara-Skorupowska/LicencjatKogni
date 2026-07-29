(define (domain epuck-arena)
  (:requirements :strips :typing)
  (:types robot)
  (:predicates 
    (obstacle-ahead ?r - robot)
    (door-ahead ?r - robot)
    (clear-path ?r - robot)
  )

  (:action turn-until-clear
    :parameters (?r - robot)
    :precondition (obstacle-ahead ?r)
    :effect (and (not (obstacle-ahead ?r)) (clear-path ?r))
  )

  (:action roam
    :parameters (?r - robot)
    :precondition (clear-path ?r)
    :effect (and (not (clear-path ?r)) (door-ahead ?r))
  )
)