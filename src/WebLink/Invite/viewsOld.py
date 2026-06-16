from django.shortcuts import render, redirect
from django.urls import reverse
from django import forms
import os
from django.conf import settings
import icalendar
from pytz import timezone
from datetime import timedelta
import json
from django.http import JsonResponse
from icalendar import Event, Calendar
from datetime import datetime, timedelta

def selectDateAndTime(request):
  ics_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'test.ics')

  if request.method == "POST":
    
    free_times = []
    selected_date = request.POST.get('selectedDate', None)
    # selected_day_time = request.POST.get('selectedDayTime')
    # print(selected_day_time)

    if selected_date:
      intervals = extractFreeTime(ics_file_path, 'America/New_York')
      
      for interval in intervals:
        if str(interval[0].date()) == selected_date:
          start_time = interval[0]
          end_time = interval[1]
          # makes sure free time intervals are 30minutes
          if start_time + timedelta(minutes=30) <= end_time:
            free_times.append({
              'start_time': start_time.strftime('%H:%M:%S'),
            })
            # print(f"Start Time: {start_time.strftime('%H:%M:%S')}, End Time: {end_time.strftime('%H:%M:%S')}")
    context = {'free_times': free_times, 'visible': True, 'date': selected_date}
    return render(request, 'Invite/select.html', context)
  
  # na_days = extractDaysWithNoAvailableTime(ics_file_path, 'America/New_York')

  noTimes = []

  # for day in na_days:
  #   date = ( f'{day.year}-{day.month}-{day.day}' )
  #   noTimes.append(date)
  
  context = {'na_days': json.dumps(noTimes)}
    

  return render(request, 'Invite/select.html', context)
  # free_times = []

  # if request.method == "POST":
  #   selected_date = request.POST.get('selectedDate', None)
  #   print
  #   if selected_date:
  #     # print(selected_date)
  #     ics_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'test.ics')
  #     intervals = extractFreeTime(ics_file_path, 'America/New_York')
      
  #     for interval in intervals:
  #       if str(interval[0].date()) == selected_date:
  #         start_time = interval[0]
  #         end_time = interval[1]
  #         # makes sure free time intervals are 30minutes
  #         if start_time + timedelta(minutes=30) <= end_time:
  #           free_times.append({
  #             'start_time': start_time.strftime('%H:%M:%S'),
  #           })
  #           # print(f"Start Time: {start_time.strftime('%H:%M:%S')}, End Time: {end_time.strftime('%H:%M:%S')}")
  #   context = {'free_times': free_times}
  #   print(context)
  #   return render(request, 'Invite/select.html', context)
  
  # context = {'free_times': free_times}
  # return render(request, 'Invite/select.html', context)


def extractFreeTime(calendar_file, timezone_name):
  with open(calendar_file, 'rb') as file:
    calendar_data = file.read()

  calendar = icalendar.Calendar.from_ical(calendar_data)

  events = []
  for component in calendar.walk():
    if component.name == 'VEVENT':
      start = component.get('dtstart').dt
      end = component.get('dtend').dt
      events.append((start, end))

  events.sort(key=lambda x: x[0])

  timezone_obj = timezone(timezone_name)
  free_time = []
  previous_end = None
  for event in events:
    start, end = event
    if previous_end and start > previous_end:
      start = start.astimezone(timezone_obj)
      end = end.astimezone(timezone_obj)
      free_time.append([start,end])
    previous_end = end

  return free_time


class InfoForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    phone = forms.CharField(max_length=10)
    topic = forms.CharField(max_length=500)


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib, ssl































def send_email(name, email, topic, time="30 min meeting", date="blank"):
  port = 465
  password = "thrmuixnmvbzerig"
  sender_email = "testtesttenet@gmail.com"
  receiver_email = email

  title = "Placeholder Title"
  start_time = datetime(2023, 9, 10, 10, 0, 0)
  duration = 30
  location = "Placeholder Location"
  link = "https://example.com"
  participants = ["saminsarker05@gmail.com", email]
  ics_content = generate_calendar_invite(title, start_time, duration, location, link, participants)

  message = MIMEMultipart()

  message['Subject'] = f"New Event: Samin Sarker - 11:00am Wed, Sep 6, 2023 - {time}"
  message['From'] = sender_email
  message['To'] = receiver_email

  # <div id="image-container">
  #         <img src="{% static 'myapp/images/your_image.jpg' %}" alt="Your Image">
  #     </div>


  html_content = f"""
  <html>
  <body>
      <div style="text-align: left; border-top: 1px black solid; width: 35%;">
          <p>Hi Ben G. Valk</p>
          <p>A new event has been scheduled.</p>

          <p>Event Type: {time}</p>

          <p>Invitee:</p>
          <p>{name}</p>

          <p>Invitee Email:</p>
          <p>{email}</p>

          <p>Event Date/time:</p>
          <p>11:00am Wed, Sep 6, 2023</p>

          <p>Invited time zone:</p>
          <p>Pacific Time US/CANADA</p>

          <p>Here is your calendar invite:</p>
      </div>
  </body>
  </html>
  """


  html_message = MIMEText(html_content, 'html')
  message.attach(html_message)

  ics_attachment = MIMEText(ics_content, 'calendar; method=REQUEST')
  ics_attachment.add_header('Content-Disposition', 'attachment', filename='calendar_invite.ics')
  message.attach(ics_attachment)

  context = ssl.create_default_context()




  try:
      with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
          server.login(sender_email, password)
          sender_mail = f'Meeting set up with {name} at {time} to discuss {topic}'
          server.sendmail(sender_email, "saminsarker05@gmail.com", message.as_string())
      print("Email sent successfully!")
  except Exception as e:
      print(f"Email could not be sent. Error: {str(e)}")



    

  try:
      with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
          server.login(sender_email, password)
          server.sendmail(sender_email, receiver_email, message.as_string())
      print("Email sent successfully!")
  except Exception as e:
      print(f"Email could not be sent. Error: {str(e)}")


# receiver_email = f"""
#   <html>
#   <body>
#       <div style="text-align: left; border-top: 1px black solid; width: 35%;">
#           <p>Hi Samin</p>
#           <p>Your 30min meeting with Ben G. Valk at 11:00 AM (Pacific Time - US and Canada) on Wednesday, September 6, 2023 is scheduled</p>

#           <p>Here is your calendar invite:</p>
#       </div>
#   </body>
#   </html>
#   """
    




def generate_calendar_invite(title, start_time, duration, location, link, participants):
    ics_content = f"""
    BEGIN:VCALENDAR
    VERSION:2.0
    PRODID:-//Example Inc.//Example Calendar//EN
    BEGIN:VEVENT
    SUMMARY:{title}
    DTSTART:{start_time.strftime('%Y%m%dT%H%M%SZ')}
    DTEND:{(start_time + timedelta(minutes=duration)).strftime('%Y%m%dT%H%M%SZ')}
    LOCATION:{location}
    DESCRIPTION:Link: {link}\\nParticipants: {", ".join(participants)}
    END:VEVENT
    END:VCALENDAR
    """
    return ics_content





# def generate_calendar_invite(title, start_time, duration, location, link, participants):
#     cal = Calendar()
#     event = Event()
#     event.add('summary', title)
#     event.add('dtstart', start_time)
    
#     end_time = start_time + timedelta(minutes=duration)
#     event.add('dtend', end_time)
    
#     event.add('location', location)
#     event.add('description', f'Link: {link}\nParticipants: {", ".join(participants)}')
    
#     for participant in participants:
#         event.add('attendee', participant)

#     cal.add_component(event)

#     with open('calendar_invite.ics', 'wb') as f:
#         f.write(cal.to_ical())








































def enterInformation(request):
  if request.method == 'POST':
    form = InfoForm(request.POST)
    if form.is_valid():
      name = form.cleaned_data['name']
      email = form.cleaned_data['email']
      phone = form.cleaned_data['phone']
      topic = form.cleaned_data['topic']
      print(name, email, phone, topic)

      send_email(name, email, topic)
      return redirect(reverse('pageThree'))
  else:
    form = InfoForm()

  return render(request, 'Invite/info.html', {'form': form})


def confirmation(request):
  return render(request, 'Invite/confirm.html')



from icalendar import Calendar
from datetime import datetime, timedelta
from pytz import timezone

# def extractDaysWithNoAvailableTime(calendar_file, timezone_name):
#     with open(calendar_file, 'rb') as file:
#         calendar_data = file.read()

#     calendar = icalendar.Calendar.from_ical(calendar_data)

#     events = []
#     for component in calendar.walk():
#         if component.name == 'VEVENT':
#             start = component.get('dtstart').dt
#             end = component.get('dtend').dt
#             events.append((start, end))

#     events.sort(key=lambda x: x[0])

#     timezone_obj = timezone(timezone_name)
#     busy_days = set()
#     previous_end = None

#     # Calculate the current date and the date 2 months from now
#     today = datetime.now(timezone_obj)
#     two_months_later = today + timedelta(days=60)

#     for event in events:
#         start, end = event
#         if previous_end and start > previous_end:
#             start = start.astimezone(timezone_obj)
#             end = end.astimezone(timezone_obj)
#             while start < end:
#                 # Check if the date is within the 2-month range
#                 if today <= start <= two_months_later:
#                     busy_days.add(start.date())
#                 start += timedelta(days=1)
#         previous_end = end

#     # Generate a list of all days within the 2-month range that are not in busy_days
#     all_days = [today + timedelta(days=i) for i in range((two_months_later - today).days + 1)]
#     days_with_no_available_time = [day for day in all_days if day.date() not in busy_days]

#     return days_with_no_available_time