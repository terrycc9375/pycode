<script setup>
import { ref, reactive, defineEmits } from 'vue'

const Iuno = {
    // 角色 + 萬物持存的注釋R1
    hp: 10525, // 基礎生命
    atk: 450 + 500, // 基礎攻擊
    atki: 0, // 額外攻擊
    atkp: 24, // 攻擊%, 天賦12% + 專武12%
    def: 1124, // 基礎防禦
    dmg_amp: 40 + 40, // 天賦 + C2
    aero_dmg: 0,
    basic_atk: 0,
    heavy_atk: 0,
    res_skill: 0,
    res_lib: 20, // 專武
    cr_rate: 49, // 5% + 8% + 36%
    cr_dmg: 150,
}

const IunoEcho = {
    // 3榮鬥 + 2風
    hp: 4560,
    atk: 450,
    atkp: 30,
    aero_dmg: 22,
    res_lib: 12,
    cr_dmg: 20,
}

const secondary = ref({
    atk: null,
    atkp: null,
    basic_atk: null,
    heavy_atk: null,
    res_skill: null,
    res_lib: null,
    cr_rate: null,
    cr_dmg: null,
})

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
    Z: new Skill("至臻的完滿", 2, 1759.05),
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

</script>

<template>
    <div class="inputArea">
        <h2>填寫聲骸</h2>

        攻擊<input v-model.number="secondary.atkp">%&ensp;+ 
        <input v-model.number="secondary.atk"><br/>
        暴擊<input v-model.number="secondary.cr_rate">% <br/>
        暴擊傷害<input v-model.number="secondary.cr_dmg">% <br/>
        普攻傷害加成<input v-model.number="secondary.basic_atk">% <br/>
        重擊傷害加成<input v-model.number="secondary.heavy_atk">% <br/>
        共鳴技能傷害加成<input v-model.number="secondary.res_skill">% <br/>
        共鳴解放加成<input v-model.number="secondary.res_lib">% <br/>
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
                    <td>{{  skill.multiplier }}%</td>
                    <td>{{  renderSkillType(skill) }}</td>
                    <td>{{  }}</td>
                    <td>{{  }}</td>
                    <td>{{  }}</td>
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
