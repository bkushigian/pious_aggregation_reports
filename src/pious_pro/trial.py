import datetime

trial_end = datetime.date(2024, 12, 31)
# if datetime.date.today() > trial_end:
if False:
    print("Trial expired. Please contact software maintainer.")
    exit(2)
