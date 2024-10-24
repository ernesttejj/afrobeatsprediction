from django.db import models

# Create your models here.
class AudioFile(models.Model):
    title = models.CharField(max_length = 255)
    audio_file = models.FileField(upload_to='audio_files/')
