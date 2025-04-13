export const metadata = {
  title: 'CV Job Matcher',
  description: 'Upload your CV to find matching job opportunities',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}