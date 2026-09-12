(define (domain world)
  (:predicates
    (active_Open_the_Door)
    (active_Finish)
    (active_Find_the_Goal)
    (active_Aproach_the_Door)
    (active_Find_the_Door)
    (active_Start)
    (active_Aproach_the_Goal)
  )

  (:action Open_the_Door
    :parameters ()
    :precondition (active_Open_the_Door)
    :effect (not (active_Open_the_Door))
  )

  (:action Finish
    :parameters ()
    :precondition (active_Finish)
    :effect (not (active_Finish))
  )

  (:action Find_the_Goal
    :parameters ()
    :precondition (active_Find_the_Goal)
    :effect (and (not (active_Find_the_Goal)) (active_Aproach_the_Door) (active_Find_the_Door) (active_Start))
  )

  (:action Aproach_the_Door
    :parameters ()
    :precondition (active_Aproach_the_Door)
    :effect (and (not (active_Aproach_the_Door)) (active_Find_the_Goal) (active_Find_the_Door) (active_Start))
  )

  (:action Find_the_Door
    :parameters ()
    :precondition (active_Find_the_Door)
    :effect (and (not (active_Find_the_Door)) (active_Find_the_Goal) (active_Aproach_the_Door) (active_Start))
  )

  (:action Start
    :parameters ()
    :precondition (active_Start)
    :effect (and (not (active_Start)) (active_Find_the_Goal) (active_Find_the_Door))
  )

  (:action Aproach_the_Goal
    :parameters ()
    :precondition (active_Aproach_the_Goal)
    :effect (and (not (active_Aproach_the_Goal)) (active_Finish))
  )
)