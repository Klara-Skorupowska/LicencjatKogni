(define (domain robot-skills)
  (:requirements :typing)
  (:types state)
  (:predicates
    (Pre-TurnAway ?s - state)
    (Eff-TurnAway ?s - state)
    (Pre-Approach ?s - state)
    (Eff-Approach ?s - state)
    (Pre-Turn ?s - state)
    (Eff-Turn ?s - state)
    (robot-at ?s - state)
  )

  (:action execute-TurnAway
    :parameters (?from - state ?to - state)
    :precondition (and (robot-at ?from) (Pre-TurnAway ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to) (Eff-TurnAway ?to))
  )
  (:action execute-Approach
    :parameters (?from - state ?to - state)
    :precondition (and (robot-at ?from) (Pre-Approach ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to) (Eff-Approach ?to))
  )
  (:action execute-Turn
    :parameters (?from - state ?to - state)
    :precondition (and (robot-at ?from) (Pre-Turn ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to) (Eff-Turn ?to))
  )
)