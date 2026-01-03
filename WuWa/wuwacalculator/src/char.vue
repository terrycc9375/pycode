<script setup>
import { ref, reactive, defineEmits, computed } from 'vue'

const Iuno = {
    // 角色 + 萬物持存的注釋R1
    hp: 10525, // 基礎生命
    atk: 450 + 500, // 基礎攻擊
    atki: 0, // 額外攻擊
    atkp: 12 + 12 + 40, // 攻擊%, 天賦12% + 專武12% + C1
    def: 1124, // 基礎防禦
    dmg_amp: 40 + 40, // 天賦 + C2
    aero_dmg: 0,
    basic_atk: 0,
    heavy_atk: 0,
    res_skill: 0,
    res_lib: 20 + 20, // 專武 + C5
    cr_rate: 49, // 5% + 8% + 36%
    cr_dmg: 150,
}

const IunoEcho = {
    // 3榮鬥 + 2風，不計算主/副詞條
    hp: 2280 * 3, // 1C*2
    atki: 150 + 100 * 2, // 4C + 3C*2
    atkp: 30, // 3榮鬥
    aero_dmg: 12 + 10 + 60, // 海之女 + 2風 + 3C*2
    res_lib: 12, // 海之女
    cr_dmg: 20, // 3榮鬥
}

const secondary = ref({
    atki: 0,
    atkp: 0,
    basic_atk: 0,
    heavy_atk: 0,
    res_skill: 0,
    res_lib: 0,
    cr_rate: 0,
    cr_dmg: 0,
})

const secondary_2025 = {
    atki: 40,
    atkp: 74,
    basic_atk: 8.6,
    heavy_atk: 0,
    res_skill: 7.9,
    res_lib: 25.8,
    cr_rate: 37.5,
    cr_dmg: 126.2,
}

class Skill {
    constructor(name, type, multiplier) {
        this.name = name
        this.type = type
        this.multiplier = multiplier
    }
}

const IunoSkills = reactive({
    MoonringBasic: new Skill("月環普攻", 0, 493.87),
    MidAir: new Skill("空中攻擊", 0, 107.36),
    MoonbowBasic: new Skill("月弓普攻", 2, 627.48),
    E1: new Skill("原初的律動", 4, 261.07),
    E2: new Skill("告終的宣響", 4, 426.46),
    E3: new Skill("未終的宣響", 4, 426.46),
    E4: new Skill("越限的弦引", 2, 439.58),
    FluxMoonbow: new Skill("流變·月弓", 2, 250.51), // ring -> bow
    FluxMoonring: new Skill("流變·月環", 2, 316.72), // bow -> ring
    MoonbowBasicEnhanced: new Skill("強化月弓普攻", 2, 1205.08),
    E4Enhanced: new Skill("強化越限的弦引", 2, 638.38),
    Z: new Skill("至臻的完滿", 2, 2703.85), // 須測試是159.05+1600還是159.05*17
    BeneathLunarTides: new Skill("溺失月海", 2, 1093.46),
})

const renderSkillType = (skill) => {
    switch(skill.type) {
        case 0: return "普攻傷害"
        case 1: return "重擊傷害"
        case 2: return "共鳴解放"
        case 3: return "變奏技能"
        case 4: return "共鳴技能傷害"
        case 5: return "聲骸技能傷害"
        default: return "未知"
    }
}

const IunoCalculator = computed(() => {
    const result = {}
    // const sec = secondary.value
    const sec = secondary_2025
    result.stats = {
        hp: Iuno.hp + IunoEcho.hp,
        atk: (Iuno.atk) * (1+(Iuno.atkp+IunoEcho.atkp+sec.atkp)/100) + (Iuno.atki+IunoEcho.atki+sec.atki),
        cr_rate: Iuno.cr_rate + sec.cr_rate,
        cr_dmg: Iuno.cr_dmg + IunoEcho.cr_dmg + sec.cr_dmg,
        res_lib: Iuno.res_lib + IunoEcho.res_lib + sec.res_lib,
    }
    result.skillDamage = {
        "月環普攻": 
        result.stats.atk 
        * IunoSkills.MoonringBasic.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.basic_atk + sec.basic_atk) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "空中攻擊":
        result.stats.atk
        * IunoSkills.MidAir.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.basic_atk + sec.basic_atk) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "月弓普攻":
        result.stats.atk
        * IunoSkills.MoonbowBasic.multiplier / 100
        * (1 + Iuno.dmg_amp / 100 + 0.65) // C3
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "原初的律動":
        result.stats.atk
        * IunoSkills.E1.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_skill + sec.res_skill) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "告終的宣響":
        result.stats.atk
        * IunoSkills.E2.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_skill + sec.res_skill) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "未終的宣響":
        result.stats.atk
        * IunoSkills.E3.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_skill + sec.res_skill) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "越限的弦引":
        result.stats.atk
        * IunoSkills.E4.multiplier / 100
        * (1 + Iuno.dmg_amp / 100 + 0.65) // C3
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "流變·月弓":
        result.stats.atk
        * IunoSkills.FluxMoonbow.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "流變·月環":
        result.stats.atk
        * IunoSkills.FluxMoonring.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "強化月弓普攻":
        result.stats.atk
        * IunoSkills.MoonbowBasicEnhanced.multiplier / 100
        * (1 + Iuno.dmg_amp / 100 + 0.65) // C3
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "強化越限的弦引":
        result.stats.atk
        * IunoSkills.E4Enhanced.multiplier / 100
        * (1 + Iuno.dmg_amp / 100 + 0.65) // C3
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "至臻的完滿":
        result.stats.atk
        * IunoSkills.Z.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
        "溺失月海":
        result.stats.atk
        * IunoSkills.BeneathLunarTides.multiplier / 100
        * (1 + Iuno.dmg_amp / 100)
        * (1 + (Iuno.aero_dmg + IunoEcho.aero_dmg + Iuno.res_lib + sec.res_lib) / 100)
        * defMultiplier(enemyLevel.value)
        * resistanceMultiplier(enemyResistance.value),
    }
    return result;
})

const enemyLevel = ref(100)
const defMultiplier = (enemyLevel) => (0.64) * 190 / (289 + enemyLevel)
const enemyResistance = ref(10)
const resistanceMultiplier = (resistance) => 1 - resistance / 100
const expected = (stats) => stats.cr_rate/100 * (stats.cr_dmg/100 - 1) + 1

</script>

<template>
    <div class="inputArea">
        <h2>填寫聲骸</h2>

        攻擊<input v-model.number="secondary.atkp">%&ensp;+ 
        <input v-model.number="secondary.atki"><br/>
        暴擊<input v-model.number="secondary.cr_rate">% <br/>
        暴擊傷害<input v-model.number="secondary.cr_dmg">% <br/>
        普攻傷害加成<input v-model.number="secondary.basic_atk">% <br/>
        重擊傷害加成<input v-model.number="secondary.heavy_atk">% <br/>
        共鳴技能傷害加成<input v-model.number="secondary.res_skill">% <br/>
        共鳴解放傷害加成<input v-model.number="secondary.res_lib">% <br/>
        敵人等級<input v-model.number="enemyLevel">&ensp;&ensp;敵人抗性<input v-model.number="enemyResistance">%
    </div>
    <div>
        <h2>面板展示</h2>
        <p>生命: {{ Math.ceil(IunoCalculator.stats?.hp) }}</p>
        <p>攻擊: {{ Math.ceil(IunoCalculator.stats?.atk) }}</p>
        <p>暴擊: {{ IunoCalculator.stats?.cr_rate }} %</p>
        <p>暴擊傷害: {{ IunoCalculator.stats?.cr_dmg }} %</p>
        <p>共鳴解放傷害加成: {{ IunoCalculator.stats?.res_lib }} %</p>
    </div>
    <div>
        <h2>傷害展示</h2>

        <table>
            <thead>
                <tr>
                    <th>技能名稱</th>
                    <th>傷害倍率</th>
                    <th>傷害類型</th>
                    <th>無暴擊</th>
                    <th>期望值</th>
                    <th>暴擊傷害</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="skill in IunoSkills" :key="skill.name">
                    <td>{{ skill.name }}</td>
                    <td>{{ skill.multiplier }}%</td>
                    <td>{{ renderSkillType(skill) }}</td>
                    <td>{{ Math.ceil(IunoCalculator.skillDamage[skill.name]) ?? 0 }}</td>
                    <td>{{ Math.ceil(IunoCalculator.skillDamage[skill.name] * expected(IunoCalculator.stats)) ?? 0 }}</td>
                    <td>{{ Math.ceil(IunoCalculator.skillDamage[skill.name] * IunoCalculator.stats.cr_dmg/100) ?? 0 }}</td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<style lang="css" scoped>
.inputArea input {
    margin-left: 1ch;
    margin-right: 1ch;
    margin-top: 8px;
    margin-bottom: 8px;
    width: 7ch;
}
table {
    width: 80%;
    border-collapse: collapse;
}
th, td {
    border: 1px solid #ddd;
    padding: 5px 0px 5px 0px;
    text-align: center;
}
</style>
