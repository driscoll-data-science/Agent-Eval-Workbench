# Suites

`example/` is the public 10-case suite that CI runs end to end with the fake agent.
Real suites live outside the repo under `$AEB_HOME/suites/<name>/` and are never committed.

Layout of a suite directory:

    suite.yaml          criteria and tasks
    fixtures/<name>/    starting workspace for each workspace task
    hidden_tests/<name>/ tests copied in at grading time only (never visible to the agent)
