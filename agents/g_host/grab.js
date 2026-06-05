fetch('https://example.com/path/to/file.pdf')
  .then(response => response.blob())
  .then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'file.pdf';
    a.click();
    URL.revokeObjectURL(url);
  })
  .catch(error => {
    console.error('File download failed:', error);
  });
