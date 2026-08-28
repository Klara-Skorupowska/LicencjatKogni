(define (problem reach-end-pad)
  (:domain robot-skills)
  (:objects
    s0 s23 - state
  )
  (:init
    (robot-at s23)
  )
  (:goal
    (robot-at s0)
  )
)