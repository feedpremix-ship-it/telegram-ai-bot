import asyncio
import json
import httpx
from .charts import isolated_render

class Tools:
    def __init__(self, cfg, client):
        self.cfg, self.client = cfg, client
        self.plot_gate = asyncio.Semaphore(1)

    async def run(self, name, args):
        try:
            if name == 'web_search':
                if not self.cfg.tavily_key:
                    return 'Поиск не настроен: нет TAVILY_API_KEY. Не выдавай знания модели за результат поиска.', None
                query = args.get('query', '')
                if not isinstance(query, str) or not 1 <= len(query) <= 500:
                    return 'Уточните поисковый запрос.', None
                r = await self.client.post('https://api.tavily.com/search', headers={'Authorization':'Bearer '+self.cfg.tavily_key},
                                           json={'query':query,'max_results':3,'search_depth':'basic','include_answer':False})
                r.raise_for_status()
                results = r.json().get('results', [])[:3]
                return json.dumps([{'title':str(x.get('title',''))[:200], 'content':str(x.get('content',''))[:2000],
                                    'url':str(x.get('url',''))[:1000]} for x in results], ensure_ascii=False), None
            if name == 'weather':
                city = args.get('city', '')
                if not isinstance(city, str) or not 1 <= len(city) <= 100:
                    return 'Нужно полное название города.', None
                params = {'name':city,'count':5,'language':'ru','format':'json'}
                country = args.get('country_code', '')
                if isinstance(country, str) and len(country) == 2:
                    params['countryCode'] = country.upper()
                r = await self.client.get('https://geocoding-api.open-meteo.com/v1/search', params=params)
                r.raise_for_status()
                places = r.json().get('results', [])
                if not places:
                    return 'Город не найден. Попроси уточнить название и страну.', None
                if len(places) > 1:
                    return 'Уточни населённый пункт: '+json.dumps([{'name':x.get('name'), 'country':x.get('country'), 'region':x.get('admin1')} for x in places], ensure_ascii=False), None
                p = places[0]
                r = await self.client.get('https://api.open-meteo.com/v1/forecast', params={'latitude':p['latitude'],'longitude':p['longitude'],
                    'current':'temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m','timezone':'auto'})
                r.raise_for_status()
                return json.dumps({'place':p['name'],'country':p.get('country'),'timezone':r.json().get('timezone'),
                                   'current':r.json()['current'],'units':r.json().get('current_units'),
                                   'attribution':'Weather data by Open-Meteo (данные модели погоды; это не измерение на месте)'}, ensure_ascii=False), None
            if name == 'plot':
                async with self.plot_gate:
                    image = await asyncio.to_thread(isolated_render, args)
                return 'График построен и будет отправлен отдельно. Не утверждай, что это генерация изображения нейросетью.', image
            return 'Неизвестный инструмент.', None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError):
            return 'Инструмент недоступен или параметры некорректны. Не придумывай результат; попроси уточнить запрос или повторить позже.', None
