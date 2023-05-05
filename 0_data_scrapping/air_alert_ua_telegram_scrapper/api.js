import path from 'path';
import MTProto from '@mtproto/core';
import {sleep} from '@mtproto/core/src/utils/common/index.js';
import {readFileSync} from 'fs';

const {api_id, api_hash} = JSON.parse(readFileSync('./config.json'));

class API {
    constructor() {
        this.mtproto = new MTProto({api_id, api_hash, storageOptions: {path: './session.json'}});
    }

    async call(method, params, options = {}) {
        try {
            const result = await this.mtproto.call(method, params, options);
            return result;
        } catch (error) {
            console.log(`${method} error:`, error);

            const {error_code, error_message} = error;

            if (error_code === 420) {
                const seconds = Number(error_message.split('FLOOD_WAIT_')[1]);
                const ms = seconds * 1000;

                await sleep(ms);

                return this.call(method, params, options);
            }

            if (error_code === 303) {
                const [type, dcIdAsString] = error_message.split('_MIGRATE_');

                const dcId = Number(dcIdAsString);

                // If auth.sendCode call on incorrect DC need change default DC, because
                // call auth.signIn on incorrect DC return PHONE_CODE_EXPIRED error
                if (type === 'PHONE') {
                    await this.mtproto.setDefaultDc(dcId);
                } else {
                    Object.assign(options, {dcId});
                }

                return this.call(method, params, options);
            }

            return Promise.reject(error);
        }
    }

    async getUser() {
        try {
            const user = await this.call('users.getFullUser', {
                id: {
                    _: 'inputUserSelf',
                },
            });

            return user;
        } catch (error) {
            return null;
        }
    }

    sendCode(phone) {
        return this.call('auth.sendCode', {
            phone_number: phone,
            settings: {
                _: 'codeSettings',
            },
        });
    }

    signIn({code, phone, phone_code_hash}) {
        return this.call('auth.signIn', {
            phone_code: code,
            phone_number: phone,
            phone_code_hash: phone_code_hash,
        });
    }

    signUp({phone, phone_code_hash}) {
        return this.call('auth.signUp', {
            phone_number: phone,
            phone_code_hash: phone_code_hash,
            first_name: 'MTProto',
            last_name: 'Core',
        });
    }

    getPassword() {
        return this.call('account.getPassword');
    }

    checkPassword({srp_id, A, M1}) {
        return this.call('auth.checkPassword', {
            password: {
                _: 'inputCheckPasswordSRP',
                srp_id,
                A,
                M1,
            },
        });
    }
}

export default new API();
