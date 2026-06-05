
import { Ai } from './vendor/@cloudflare/ai.js';

import { escapeHtml } from './escape.js';
import template from './template.html';

export default {

// all right here we go, this might be stupid but...
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⢔⣒⠂⣀⣀⣤⣄⣀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⣴⣿⠋⢠⣟⡼⣷⠼⣆⣼⢇⣿⣄⠱⣄
//⠀⠀⠀⠀⠀⠀⠀⠹⣿⡀⣆⠙⠢⠐⠉⠉⣴⣾⣽⢟⡰⠃
//⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣦⠀⠤⢴⣿⠿⢋⣴⡏⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡙⠻⣿⣶⣦⣭⣉⠁⣿⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠀⠈⠉⠉⠉⠉⠇⡟⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⣘⣦⣀⠀⠀⣀⡴⠊⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠛⢻⣿⣿⣿⣿⠻⣧⡀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠫⣿⠉⠻⣇⠘⠓⠂⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⢶⣾⣿⣿⣿⣿⣿⣶⣄⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣧⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠈⠙⠻⢿⣿⣿⠿⠛⣄⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡁⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣷⠂⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⡀⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠀⠀⠀
//⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀for Mother Medusa, Meteora, Minotaura

  async fetch(request, env) {

    // 
    const tasks = [];
    const ai = new Ai(env.AI);

    // prompt - simple completion style input
    let simple = {
      prompt: 'Tell me a joke about Cloudflare'
    };
    let response = await ai.run('@cf/meta/llama-2-7b-chat-int8', simple);

    console.log(response)

    console.log(`${response}`) // this causes the objObj issue
    //return response; 
    //return Response.json(tasks);
    //




    const defaultData = { todos: [] };

    const setCache = (key, data) => env.EXAMPLE_TODOS.put(key, data);
    const getCache = key => env.EXAMPLE_TODOS.get(key);

    //const ai = new Ai(env.AI);

    async function rinseDream(dream){
      //const tasks = [];
      
      console.log(`ai call with %message: ${dream}`);

      // prompt - simple completion style input
      let simple = {
        prompt: `Embellish this dream with some clever emoji: ${dream}`
      };
      let response = await ai.run('@cf/meta/llama-2-7b-chat-int8', simple);
      console.log(`response: ${response}`);
      //console.log(JSON.stringify(response));

      return response;
      

      //tasks.push({ inputs: simple, response });
  
      // messages - chat style input
      /*let chat = {
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Who won the world series in 2020?' }
        ]
      };
      response = await ai.run('@cf/meta/llama-2-7b-chat-int8', chat);
      tasks.push({ inputs: chat, response });*/
  
      //return Response.json(tasks);


    }


    

    async function getTodos(request) {
      const ip = request.headers.get('CF-Connecting-IP');
      const cacheKey = `data-${ip}`;
      let data;
      const cache = await getCache(cacheKey);
      if (!cache) {
        await setCache(cacheKey, JSON.stringify(defaultData));
        data = defaultData;
      } else {
        data = JSON.parse(cache);
      }


      // Get all keys from Cloudflare KV
      //const keys = await getAllKeys();



      const body = template.replace(
        '$TODOS',
        JSON.stringify(


          // refactor into standard message format later 

          data.todos?.map(todo => ({
            id: escapeHtml(todo.id),
            name: escapeHtml(todo.name),
            completed: !!todo.completed
          })) ?? []
        )
      );

      return new Response(body, {
        headers: { 'Content-Type': 'text/html' }
      });
    }

    async function updateTodos(request) {
      const body = await request.text();

      // send to dreamLLM
      const proc_body = await rinseDream(body);
      

      const ip = request.headers.get('CF-Connecting-IP');

          const urlToFetch = 'https://github.com/greydoubt';  // Replace with the desired URL
    const webResponse = await fetch(urlToFetch);
    const webData = await webResponse.text(); 



      const cacheKey = `data-${ip}`;
      try {
        JSON.parse(proc_body);
        
        //await setCache(cacheKey, body);
        await setCache(cacheKey, proc_body);


        return new Response(proc_body, { status: 200 });
      } catch (err) {
        return new Response(err, { status: 500 });
      }
    }

    if (request.method === 'PUT') {
      return updateTodos(request);
    }
    return getTodos(request);
  }
};
