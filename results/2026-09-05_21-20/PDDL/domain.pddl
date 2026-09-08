(define (domain continuous_world)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (at ?loc)
    (can_GoThroughTheDoor ?from ?to)
    (can_GoToTheDoor ?from ?to)
    (can_GoToTheGoal ?from ?to)
    (can_SpotTheDoor ?from ?to)
    (can_SpotTheGoal ?from ?to)
  )

  (:action GoThroughTheDoor
    :parameters (?from ?to)
    :precondition (and (at ?from) (can_GoThroughTheDoor ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )

  (:action GoToTheDoor
    :parameters (?from ?to)
    :precondition (and (at ?from) (can_GoToTheDoor ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )

  (:action GoToTheGoal
    :parameters (?from ?to)
    :precondition (and (at ?from) (can_GoToTheGoal ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )

  (:action SpotTheDoor
    :parameters (?from ?to)
    :precondition (and (at ?from) (can_SpotTheDoor ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )

  (:action SpotTheGoal
    :parameters (?from ?to)
    :precondition (and (at ?from) (can_SpotTheGoal ?from ?to))
    :effect (and (at ?to) (not (at ?from)))
  )

)