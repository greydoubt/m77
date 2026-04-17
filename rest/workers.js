const response = await env.AI.run('anthropic/claude-opus-4-6',{
input: 'What is Cloudflare?',
}, {
gateway: { id: "default" },
});


const response = await env.AI.run('@cf/moonshotai/kimi-k2.5',
      {
prompt: 'What is AI Gateway?'
      },
      {
metadata: { "teamId": "AI", "userId": 12345 }
      }
    );
