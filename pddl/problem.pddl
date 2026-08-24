(define (problem reach-end-pad)
  (:domain robot-skills)
  (:objects
    s0 s41 s27 - state
  )
  (:init
    (robot-at s0)
    (Pre-Turn s41)
    (Pre-Turn s41)
    (Eff-Turn s41)
    (Eff-Turn s41)
    (Eff-Turn s41)
  )
  (:goal
    (robot-at s0)
  )
)