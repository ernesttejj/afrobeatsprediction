from django import forms

class UploadAudioForm(forms.Form):
    title = forms.CharField(max_length=255)
    audio_file = forms.FileField()