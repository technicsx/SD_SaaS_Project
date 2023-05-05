import {readFileSync, writeFileSync} from 'fs';
import prompt from 'prompt';
import api from './api.js';

const user = await api.getUser();
console.log(user)
const existingMessages = JSON.parse(readFileSync('./messages.json'));
// const lastDate = existingMessages[0].date;
const lastDate = null;

if (!user) {
    const {phone} = await prompt.get('phone');
    const {phone_code_hash} = await api.sendCode(phone);

    const {code} = await prompt.get('code');
    try {
        const signInResult = await api.signIn({
            code,
            phone,
            phone_code_hash,
        });

        if (signInResult._ === 'auth.authorizationSignUpRequired') {
            const singUpResult = await api.signUp({
                phone,
                phone_code_hash,
            });
        }

        const newUser = await api.getUser();
    } catch (error) {
        if (error.error_message !== 'SESSION_PASSWORD_NEEDED') {
            console.log(`error:`, error);
        }
    }
}

const resolvedPeer = await api.call('contacts.resolveUsername', {
    username: 'air_alert_ua',
});

const channel = resolvedPeer.chats.find((chat) => chat.id === resolvedPeer.peer.channel_id);

const inputPeer = {
    _: 'inputPeerChannel',
    channel_id: channel.id,
    access_hash: channel.access_hash,
};

const firstHistoryResult = await api.call('messages.getHistory', {
    peer: inputPeer,
    limit: 1,
});
const totalCount = firstHistoryResult.count;

const allMessages = [];
const LIMIT_COUNT = 100;

for (let offset = 0; offset < totalCount; offset += LIMIT_COUNT) {
    console.log(`downloading ${offset}–${offset + LIMIT_COUNT} of ${totalCount} messages`);

    const history = await api.call('messages.getHistory', {
        peer: inputPeer,
        add_offset: offset,
        limit: LIMIT_COUNT,
    });
    const messages = history.messages.map(({message, date}) => ({message, date}));

    allMessages.push(...messages);

    const oldestDownloaded = messages[messages.length - 1];

    if (oldestDownloaded.date < lastDate) {
        for (const message of existingMessages) {
            if (message.date < oldestDownloaded.date || (message.date === oldestDownloaded.date && message.message !== oldestDownloaded.message)) {
                allMessages.push(message);
            }
        }
        break;
    }
}

writeFileSync('./messages.json', '[\n' + allMessages.map(m => JSON.stringify(m)).join(',\n') + '\n]\n');

allMessages.sort((a, b) => a.date - b.date);

const ranges = {};
const lastTypes = {};
const durations = {};

const TYPES = {
    'Повітряна тривога': 0,
    'Відбій тривоги': 1,
    'Відбій повітряної тривоги': 1
};

const re = /. \d\d:\d\d  ?(?<type>Повітряна тривога|Загроза артобстрілу|Відбій( повітряної)? тривоги|Відбій загрози артобстрілу)( в (?<location>.+?)\.?|\.|!)$/m;

const dateMin = 1647295200;
const dateMax = allMessages[allMessages.length - 1].date;

for (const {message, date} of allMessages) {
    if (!message) continue;

    const match = message.match(re);
    if (!match) {
        console.log(message);
        continue;
    }
    const tag = message.split('#')[1];
    const type = TYPES[match.groups.type];
    let location = match.groups.location;
    if (!location) {
        location = tag.replace('м_', 'м. ').replace(/_/g, ' ');
    }
    const locations = location.split(' та ');

    for (const loc of locations) {
        let lastType = lastTypes[loc];
        const alerts = (ranges[loc] = ranges[loc] || []);

        // skip stops that don't have starts
        if (alerts.length === 0 && type === 1) continue;

        if (type === lastType) continue; // skip duplicate events

        lastTypes[loc] = type;
        alerts.push(date);
    }
}

for (const loc of Object.keys(ranges)) {
    if (ranges[loc].length % 2 === 1) {
        ranges[loc].push(dateMax); // extend starts that don't have a stop to the end
    }
    if (ranges[loc].length === 0) {
        delete ranges[loc];
        continue;
    }
    for (let i = 0; i < ranges[loc].length; i += 2) {
        durations[loc] = (durations[loc] || 0) + ranges[loc][i + 1] - ranges[loc][i];
    }
    if (durations[loc] < 100) {
        delete ranges[loc];
    }
}

// sort by total siren duration
const locations = Object.keys(ranges).sort((a, b) => durations[b] - durations[a]);

const sorted = {};
for (const loc of locations) {
    // console.log(durations[loc], ranges[loc].length, loc);
    sorted[loc] = ranges[loc];
}

writeFileSync('./sirens.json', JSON.stringify({
    locations: sorted,
    updated: Math.round(Date.now() / 1000)
}));

process.exit(0);
