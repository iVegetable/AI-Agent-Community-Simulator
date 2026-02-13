/*
  Root layout shell for the simulator frontend.
  It only applies global styles and wraps the single-page experience.
*/

import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
