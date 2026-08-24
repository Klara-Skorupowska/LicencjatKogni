(define (domain robot-skills)
  (:requirements :typing)
  (:types state)
  (:predicates
    (Pre-Turn ?s - state)
    (Eff-Turn ?s - state)
    (robot-at ?s - state)
  )

  (:action execute-Turn
    :parameters (?from - state ?to - state)
    :precondition (and (robot-at ?from) (Pre-Turn ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to) (Eff-Turn ?to))
  )
)