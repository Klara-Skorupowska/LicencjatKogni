(define (domain robot_domain)
  (:requirements :strips)
  (:predicates
    (can_run_Open_the_Door)
    (executed_Open_the_Door)
    (can_run_Start)
    (executed_Start)
    (can_run_Finish)
    (executed_Finish)
    (can_run_Go_to_the_Goal)
    (executed_Go_to_the_Goal)
    (can_run_Go_to_the_Door)
    (executed_Go_to_the_Door)
    (can_run_Find_the_Door)
    (executed_Find_the_Door)
    (can_run_Find_the_Goal)
    (executed_Find_the_Goal)
  )

  (:action Open_the_Door
    :parameters ()
    :precondition (can_run_Open_the_Door)
    :effect (and (executed_Open_the_Door) (can_run_Find_the_Door) (can_run_Find_the_Goal) (not (can_run_Open_the_Door)))
  )

  (:action Start
    :parameters ()
    :precondition (can_run_Start)
    :effect (and (executed_Start) (can_run_Find_the_Door) (not (can_run_Start)))
  )

  (:action Finish
    :parameters ()
    :precondition (can_run_Finish)
    :effect (and (executed_Finish) (not (can_run_Finish)))
  )

  (:action Go_to_the_Goal
    :parameters ()
    :precondition (can_run_Go_to_the_Goal)
    :effect (and (executed_Go_to_the_Goal) (can_run_Finish) (can_run_Find_the_Door) (can_run_Find_the_Goal) (not (can_run_Go_to_the_Goal)))
  )

  (:action Go_to_the_Door
    :parameters ()
    :precondition (can_run_Go_to_the_Door)
    :effect (and (executed_Go_to_the_Door) (can_run_Open_the_Door) (can_run_Find_the_Door) (not (can_run_Go_to_the_Door)))
  )

  (:action Find_the_Door
    :parameters ()
    :precondition (can_run_Find_the_Door)
    :effect (and (executed_Find_the_Door) (can_run_Go_to_the_Door) (not (can_run_Find_the_Door)))
  )

  (:action Find_the_Goal
    :parameters ()
    :precondition (can_run_Find_the_Goal)
    :effect (and (executed_Find_the_Goal) (can_run_Go_to_the_Goal) (not (can_run_Find_the_Goal)))
  )
)