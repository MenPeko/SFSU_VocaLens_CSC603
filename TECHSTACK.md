 Runtime & Backend                                                                                                                    
                                                                                                                                       
  - Bun — runtime, package manager, and dev server (with HMR)                                                                          
  - Express (via @mentra/sdk AppServer) — handles MentraOS session/webhook routes on port 3333                                         
  - Bun serve() — serves the webview + custom API routes on port 3334, with Express proxying unmatched requests to it                  
  - SSE (Server-Sent Events) — for real-time streaming to the frontend                                                                 
                                                                                                                                       
  Frontend (Webview)                                                                                                                   
                                                                                                                                       
  - React 19 with JSX                                                                                                                
  - Tailwind CSS v4 (via bun-plugin-tailwind)
  - Radix UI — primitives for select, label, slot                                                                                      
  - shadcn/ui pattern — class-variance-authority, clsx, tailwind-merge (UI components in components/ui/)                               
  - Lucide React — icons                                                                                                               
  - tw-animate-css — animation utilities                                                                                               
                                                                                                                                       
  AI/Services                                                                                                                        
                                                                                                                                       
  - Gemini — OCR (gemini-ocr.ts) and translation (gemini-translate.ts)                                                                 
  
  MentraOS Integration                                                                                                                 
                                                                                                                                     
  - @mentra/sdk — workspace dependency for AppServer, session management, and glasses communication                                    
  
  Language & Tooling                                                                                                                   
                                                                                                                                     
  - TypeScript 5 (strict mode, ESNext target)                                                                                          
  - Bun test for testing                                                                                                             
                                                                                                                                       
  Architecture                                                                                                                       
                                                                                                                                       
  Two-server setup: Express (port 3333) handles MentraOS protocol + SSE streaming, while Bun (port 3334) serves the React webview and  
  custom API routes. Express proxies non-MentraOS requests to Bun.