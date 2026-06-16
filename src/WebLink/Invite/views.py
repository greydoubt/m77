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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib, ssl



def save_selected_time(request):
  if request.method == 'POST':
    selected_time = request.POST.get('selected_time')
    TIME = selected_time

    request.session['selected_time'] = selected_time

    return JsonResponse({'message': 'Selected time saved successfully.'})
  else:
    return JsonResponse({'message': 'Invalid request method.'}, status=400)



def selectDateAndTime(request):
  ics_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'test.ics')
  if request.method == "POST":
    free_times = []
    selected_date = request.POST.get('selectedDate', None)

    request.session['selected_date'] = selected_date
    DATE = selected_date

    selected_timezone = request.GET.get('timezone', 'America/New_York')
    request.session['selected_timezone'] = selected_timezone

    if selected_date:
      intervals = extractFreeTime(ics_file_path, selected_timezone)
      
      for interval in intervals:
        if str(interval[0].date()) == selected_date:
          start_time = interval[0]
          end_time = interval[1]

          if start_time + timedelta(minutes=30) <= end_time:
            formatted_start_time = start_time.strftime('%I:%M %p')
            free_times.append({
              'start_time': formatted_start_time,
            })

    context = {'free_times': free_times, 'visible': True, 'date': selected_date, 'selected_timezone': selected_timezone}
    return render(request, 'Invite/select.html', context)

  noTimes = []
  
  context = {'na_days': json.dumps(noTimes)}
    
  return render(request, 'Invite/select.html', context)



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



def send_email(request, name, email, topic, time="30 min meeting", sender_name = "Ben G. Valk"):

  sender_name = sender_name
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

  TIME = request.session.get('selected_time', "time_error")
  DATE = request.session.get('selected_date', "date_error")
  TIMEZONE = request.session.get('selected_timezone', "timezone_error")

  input_date = datetime.strptime(DATE, '%Y-%m-%d')
    
  DATE = input_date.strftime('%A, %b %d, %Y')

  message['Subject'] = f"New Event: {name} - {TIME} {TIMEZONE} {DATE} - {time}"
  message['From'] = sender_email
  message['To'] = receiver_email

  html_content = f"""
  <html>
  <body>
      <div style="text-align: left; border-top: 1px black solid; width: 35%;">

          <p>Hi {sender_name}</p>
          <p>A new event has been scheduled.</p>

          <p>Event Type: {time}</p>

          <div>
          <p style="font-weight: bold;">Invitee:</p>
          <p>{name}</p>
          </div>


          <div>
          <p style="font-weight: bold;">Invitee Email:</p>
          <p>{email}</p>
          </div>


          <div>
          <p style="font-weight: bold;">Event Date/time:</p>
          <p>{TIME} {DATE}</p>
          </div>

          <div>
          <p style="font-weight: bold;">Invited time zone:</p>
          <p>{TIMEZONE}</p>
          </div>

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

  message = MIMEMultipart()

  TIME = request.session.get('selected_time', "time_error")
  DATE = request.session.get('selected_date', "date_error")
  TIMEZONE = request.session.get('selected_timezone', "timezone_error")

  input_date = datetime.strptime(DATE, '%Y-%m-%d')
    
  DATE = input_date.strftime('%A, %b %d, %Y')

  message['Subject'] = f"Confirmed: {name} - {TIME} {TIMEZONE} {DATE} - {time}"
  message['From'] = sender_email
  message['To'] = receiver_email

  html_content = f"""
  <html>
  <body>
      <div style="text-align: left; border-top: 1px black solid; width: 35%;">
          <p>Hi Samin</p>
          <p>Your {time} meeting with {sender_name} at {TIME} {TIMEZONE} on {DATE} is scheduled</p>

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
          server.sendmail(sender_email, receiver_email, message.as_string())
      print("Email sent successfully!")
  except Exception as e:
      print(f"Email could not be sent. Error: {str(e)}")



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



def enterInformation(request):
  if request.method == 'POST':
    form = InfoForm(request.POST)
    if form.is_valid():
      name = form.cleaned_data['name']
      email = form.cleaned_data['email']
      phone = form.cleaned_data['phone']
      topic = form.cleaned_data['topic']
      print(name, email, phone, topic)

      send_email(request, name, email, topic)
      return redirect(reverse('pageThree'))
  else:
    form = InfoForm()

  TIME = request.session.get('selected_time', "time_error")
  DATE = request.session.get('selected_date', "date_error")
  TIMEZONE = request.session.get('selected_timezone', "timezone_error")

  input_date = datetime.strptime(DATE, '%Y-%m-%d')
    
  DATE = input_date.strftime('%A, %b %d, %Y')

  precedent = ""
  if TIMEZONE == "America/Los_Angeles":
    precedent = "PST"
  elif TIMEZONE == "America/New_York":
    precedent = "EST"
  elif TIMEZONE == "America/Chicago":
    precedent = "CST"
  elif TIMEZONE == "America/Denver":
    precedent = "MT"

  return render(request, 'Invite/info.html', {'form': form, 'time': TIME, 'date': DATE, 'precedent': precedent})



def confirmation(request):
  return render(request, 'Invite/confirm.html')
